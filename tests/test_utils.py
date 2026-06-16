import pytest
import torch

from areal.api.cli_args import MicroBatchSpec
from areal.utils.data import (
    pack_tensor_dict,
    pad_and_stack_tensors_along_first_dim,
    pad_mb_list,
    pad_sequences_to_tensors,
    reorder_list,
    split_padded_tensor_dict_into_mb_list,
    unpack_sequence,
)

BS = 16
MAX_ANSWER_LEN = 16
MAX_PROMPT_LEN = 8
VOCAB_SIZE = 100


@pytest.fixture
def mock_padded_data():
    prompt_lens = torch.randint(1, MAX_PROMPT_LEN, size=(BS,))
    answer_lens = torch.randint(1, MAX_ANSWER_LEN, size=(BS,))
    all_data = []
    for prompt_len, ans_len in zip(prompt_lens, answer_lens):
        prompt_len = int(prompt_len)
        ans_len = int(ans_len)
        seq = dict(
            input_ids=torch.randint(0, VOCAB_SIZE, size=(prompt_len + ans_len,)),
            loss_mask=torch.tensor([0] * prompt_len + [1] * ans_len),
            logprobs=torch.randn(prompt_len + ans_len),
            position_ids=torch.arange(prompt_len + ans_len),
        )
        all_data.append(seq)
    return pad_sequences_to_tensors(all_data)


@pytest.mark.parametrize("max_tokens_per_mb", [24, 36, 48, 100])
@pytest.mark.parametrize("n_mbs", [1, 2, 4, 8])
@pytest.mark.parametrize("n_mbs_divisor", [1, 2, 3])
def test_micro_batch_split(mock_padded_data, n_mbs, max_tokens_per_mb, n_mbs_divisor):
    mb_spec = MicroBatchSpec(
        n_mbs=n_mbs, max_tokens_per_mb=max_tokens_per_mb, n_mbs_divisor=n_mbs_divisor
    )

    # Unpad and split to microbatches
    packed_data = pack_tensor_dict(mock_padded_data)
    original_lens = packed_data["cu_seqlens"][1:] - packed_data["cu_seqlens"][:-1]
    assert torch.allclose(
        original_lens.long(), mock_padded_data["attention_mask"].sum(1)
    )
    split_result = split_padded_tensor_dict_into_mb_list(mock_padded_data, mb_spec)
    split_result.mbs = [pack_tensor_dict(mb) for mb in split_result.mbs]
    reordered_lens = [original_lens[i] for i in split_result.forward_indices]

    # assert microbatch split result does not violate requirements
    assert len(split_result.mbs) >= n_mbs
    assert len(split_result.mbs) % n_mbs_divisor == 0

    # test reorder back
    for key in split_result.mbs[0].keys():
        if key in ["cu_seqlens", "max_seqlen"]:
            continue

        # assert microbatch split result does not violate requirements
        for mb in split_result.mbs:
            assert mb[key].shape[0] <= max_tokens_per_mb

        x = torch.cat([mb[key] for mb in split_result.mbs])
        xs = unpack_sequence(x, lens=reordered_lens)
        xs = reorder_list(xs, split_result.backward_indices)
        x = torch.cat(xs)
        assert torch.allclose(x, packed_data[key])
        y = pad_and_stack_tensors_along_first_dim(xs)
        assert torch.allclose(mock_padded_data[key], y)


def test_synced_micro_batch_split_appends_dummy_when_rank_has_fewer_groups(
    monkeypatch,
):
    data = pad_sequences_to_tensors(
        [
            {
                "input_ids": torch.tensor([1, 2, 3]),
                "loss_mask": torch.tensor([1, 1, 1]),
                "logprobs": torch.randn(3),
            },
            {
                "input_ids": torch.tensor([4, 5]),
                "loss_mask": torch.tensor([1, 1]),
                "logprobs": torch.randn(2),
            },
        ]
    )

    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group=None: 3)

    def _mock_all_gather_object(output, obj, group=None):
        del obj, group
        output[:] = [1, 3, 2]

    monkeypatch.setattr(torch.distributed, "all_gather_object", _mock_all_gather_object)

    mb_list = split_padded_tensor_dict_into_mb_list(
        data,
        MicroBatchSpec(n_mbs=1, max_tokens_per_mb=1024),
        group=object(),
        allow_dummy_mbs=True,
    )

    assert len(mb_list.mbs) == 3
    assert mb_list.is_dummy == [False, True, True]
    assert mb_list.forward_indices == [0, 1]
    assert mb_list.backward_indices == [0, 1]

    packed_dummy = pack_tensor_dict(mb_list.mbs[-1])
    assert int(packed_dummy["cu_seqlens"][-1]) == 1
    assert int(packed_dummy["loss_mask"].count_nonzero()) == 0

    mb_list.mbs = [pack_tensor_dict(mb) for mb in mb_list.mbs]
    mb_list = pad_mb_list(mb_list, pad_to_maximum=True)
    assert [mb_item.is_dummy for mb_item in mb_list] == [False, True, True]
    assert mb_list.padded_to_lengths == [1024, 256, 256]


def test_synced_micro_batch_split_old_behavior_still_errors_without_dummy(
    monkeypatch,
):
    data = pad_sequences_to_tensors(
        [
            {
                "input_ids": torch.tensor([1, 2, 3]),
                "loss_mask": torch.tensor([1, 1, 1]),
            },
            {
                "input_ids": torch.tensor([4, 5]),
                "loss_mask": torch.tensor([1, 1]),
            },
        ]
    )

    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group=None: 2)

    def _mock_all_gather_object(output, obj, group=None):
        del obj, group
        output[:] = [1, 3]

    monkeypatch.setattr(torch.distributed, "all_gather_object", _mock_all_gather_object)

    with pytest.raises(RuntimeError, match="smaller than min_groups 3"):
        split_padded_tensor_dict_into_mb_list(
            data,
            MicroBatchSpec(n_mbs=1, max_tokens_per_mb=1024),
            group=object(),
            allow_dummy_mbs=False,
        )

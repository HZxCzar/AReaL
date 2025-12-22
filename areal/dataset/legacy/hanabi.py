from datasets import Split, load_dataset
from datasets.distributed import split_dataset_by_node


def get_hanabi_sft_dataset(path, split, tokenizer, rank, world_size):
    if tokenizer is None:
        raise ValueError("Tokenizer must be provided for Hanabi SFT dataset loading.")

    if split == "train":
        dataset = load_dataset("json", data_files=path, split="train[:95%]")
    else:
        dataset = load_dataset("json", data_files=path, split="train[95%:]")
    dataset = dataset.filter(
        lambda sample: bool(sample.get("prompt")) and bool(sample.get("response"))
    )
    dataset = split_dataset_by_node(dataset, rank=rank, world_size=world_size)

    def process(sample):
        prompt_text = sample["prompt"]      # already includes chat template + gen prompt
        response_text = sample["response"]  # assistant continuation only

        # tokenize prompt only
        prompt_ids = tokenizer(
            prompt_text,
            add_special_tokens=False,
        )["input_ids"]

        # tokenize full sequence (prompt + response)
        resp_ids = tokenizer(
            response_text,
            add_special_tokens=False,
        )["input_ids"]

        full_ids = prompt_ids + resp_ids

        # loss mask: only train on response tokens
        loss_mask = [0] * len(prompt_ids) + [1] * (len(resp_ids))

        return {
            "input_ids": full_ids,
            "loss_mask": loss_mask,
        }

    dataset = dataset.map(process)
    dataset = dataset.remove_columns(
        [col for col in dataset.column_names if col not in ["input_ids", "loss_mask"]]
    )
    return dataset


def get_hanabi_rl_dataset(path, split, rank, world_size):
    """
    Load hanabi prompts for RL training.

    Note that this dataset is actually a placeholder. The only use is to decide the length
    of training for hanabi experiment.

    Each row in the JSONL dataset should contain:
        {
            "id": str,
            "prompt": str,
      }
    """
    dataset = load_dataset("json", data_files=path, split=split)
    dataset = split_dataset_by_node(dataset, rank=rank, world_size=world_size)

    def process(sample):
        message = [{"role": "user", "content": sample["prompt"]}]
        res = {"messages": message}
        if "id" in sample:
            res["query_id"] = sample["id"]
        return res

    dataset = dataset.map(process).remove_columns([col for col in dataset.column_names if col not in ["messages", "query_id"]])
    return dataset

from datasets import load_dataset


def get_hanabi_sft_dataset(
    path: str,
    split: str,
    tokenizer,
    max_length: int | None = None,
):
    if tokenizer is None:
        raise ValueError("Tokenizer must be provided for Hanabi SFT dataset loading.")

    if split == "train":
        dataset = load_dataset("json", data_files=path, split="train[:95%]")
    else:
        dataset = load_dataset("json", data_files=path, split="train[95%:]")
    dataset = dataset.filter(
        lambda sample: bool(sample.get("prompt")) and bool(sample.get("response"))
    )

    def process(sample):
        prompt_text = sample["prompt"]  # already includes chat template + gen prompt
        response_text = sample["response"]  # assistant continuation only

        prompt_ids = tokenizer(
            prompt_text,
            add_special_tokens=False,
        )["input_ids"]

        resp_ids = tokenizer(
            response_text,
            add_special_tokens=False,
        )["input_ids"]

        full_ids = prompt_ids + resp_ids
        loss_mask = [0] * len(prompt_ids) + [1] * len(resp_ids)

        return {
            "input_ids": full_ids,
            "loss_mask": loss_mask,
        }

    dataset = dataset.map(process).remove_columns(
        [col for col in dataset.column_names if col not in ["input_ids", "loss_mask"]]
    )

    if max_length is not None:
        dataset = dataset.filter(lambda sample: len(sample["input_ids"]) <= max_length)

    return dataset


def get_hanabi_rl_dataset(
    path: str,
    split: str,
    tokenizer=None,
    max_length: int | None = None,
):
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

    def process(sample):
        message = [{"role": "user", "content": sample["prompt"]}]
        res = {"messages": message}
        if "id" in sample:
            res["query_id"] = sample["id"]
        return res

    dataset = dataset.map(process).remove_columns(
        [col for col in dataset.column_names if col not in ["messages", "query_id"]]
    )

    if max_length is not None:

        def filter_length(sample):
            content = sample["messages"][0]["content"]
            tokens = tokenizer.encode(content) if tokenizer is not None else []
            return len(tokens) <= max_length

        dataset = dataset.filter(filter_length)

    return dataset
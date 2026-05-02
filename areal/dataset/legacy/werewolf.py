from datasets import load_dataset


def get_werewolf_rl_dataset(
    path: str,
    split: str,
    tokenizer=None,
    max_length: int | None = None,
):
    """
    Load werewolf prompts for RL training.

    Note that this dataset is actually a placeholder. The only use is to decide the length
    of training for werewolf experiment.

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
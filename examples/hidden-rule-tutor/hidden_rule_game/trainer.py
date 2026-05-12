from __future__ import annotations

import argparse
from pathlib import Path

from datasets import load_dataset
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from .inference_engine import resolve_model_path


def format_sample(sample: dict[str, object]) -> str:
    return (
        "You are learning to infer hidden rules from labeled string examples.\n\n"
        f"{sample['prompt']}\n\n"
        f"{sample['answer']}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/lora-student"))
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--init-lora-path", type=Path, default=None)
    return parser.parse_args()


def train_lora(args: argparse.Namespace) -> None:
    dataset = load_dataset("json", data_files=str(args.train_file), split="train")
    dataset = dataset.map(lambda row: {"text": format_sample(row)})

    model_path = resolve_model_path(args.model)
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize(row: dict[str, object]) -> dict[str, object]:
        return tokenizer(
            str(row["text"]),
            truncation=True,
            max_length=args.max_seq_length,
            padding=False,
        )

    tokenized = dataset.map(tokenize, remove_columns=dataset.column_names)

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=args.trust_remote_code,
    )

    if args.init_lora_path:
        model = PeftModel.from_pretrained(model, str(args.init_lora_path), is_trainable=True)
    else:
        peft_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
        model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=str(args.out_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        bf16=True,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        train_dataset=tokenized,
        args=training_args,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )
    trainer.train()
    trainer.save_model(str(args.out_dir))
    tokenizer.save_pretrained(str(args.out_dir))
    print(f"Saved LoRA adapter to {args.out_dir}")


def main() -> None:
    train_lora(parse_args())


if __name__ == "__main__":
    main()

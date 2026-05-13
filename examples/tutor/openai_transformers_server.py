from __future__ import annotations

import argparse
import asyncio
import time
from typing import Any

import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "default"
    messages: list[ChatMessage]
    temperature: float = 0.7
    top_p: float = 0.95
    max_tokens: int = 512


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=30001)
    parser.add_argument("--served-model-name", default="default")
    parser.add_argument("--dtype", default="bfloat16", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def torch_dtype(name: str) -> torch.dtype | str:
    if name == "auto":
        return "auto"
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def build_prompt(tokenizer: Any, messages: list[ChatMessage]) -> str:
    payload = [{"role": msg.role, "content": msg.content} for msg in messages]
    if getattr(tokenizer, "chat_template", None):
        try:
            return tokenizer.apply_chat_template(
                payload,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            return tokenizer.apply_chat_template(payload, tokenize=False, add_generation_prompt=True)
    return "\n".join(f"{msg.role}: {msg.content}" for msg in messages) + "\nassistant:"


def create_app(args: argparse.Namespace) -> FastAPI:
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=args.trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch_dtype(args.dtype),
        device_map="auto",
        trust_remote_code=args.trust_remote_code,
    )
    model.eval()
    lock = asyncio.Lock()
    app = FastAPI()

    @app.get("/v1/models")
    async def models() -> dict[str, Any]:
        return {
            "object": "list",
            "data": [
                {
                    "id": args.served_model_name,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "local",
                }
            ],
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(req: ChatCompletionRequest) -> dict[str, Any]:
        async with lock:
            text = await asyncio.to_thread(generate, tokenizer, model, req)
        return {
            "id": f"chatcmpl-{int(time.time() * 1000)}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": args.served_model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }
            ],
        }

    return app


def generate(tokenizer: Any, model: Any, req: ChatCompletionRequest) -> str:
    prompt = build_prompt(tokenizer, req.messages)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    do_sample = req.temperature > 0
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=req.max_tokens,
            do_sample=do_sample,
            temperature=max(req.temperature, 1e-5) if do_sample else None,
            top_p=req.top_p if do_sample else None,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = output_ids[0, inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def main() -> None:
    args = parse_args()
    app = create_app(args)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()

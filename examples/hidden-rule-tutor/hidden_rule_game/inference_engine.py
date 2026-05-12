from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


@dataclass
class GenerationConfig:
    max_tokens: int = 192
    temperature: float = 0.2
    top_p: float = 0.95


class VLLMEngine:
    def __init__(
        self,
        model: str,
        tensor_parallel_size: int = 1,
        dtype: str = "auto",
        lora_path: str | None = None,
        lora_name: str = "student_lora",
        trust_remote_code: bool = False,
        download_dir: str | None = None,
        served_model_name: str | None = None,
    ) -> None:
        from vllm import LLM

        self.model = resolve_model_path(model)
        self.lora_path = str(Path(lora_path).expanduser().resolve()) if lora_path else None
        self.lora_name = lora_name
        self.llm = LLM(
            model=self.model,
            tensor_parallel_size=tensor_parallel_size,
            dtype=dtype,
            enable_lora=lora_path is not None,
            trust_remote_code=trust_remote_code,
            download_dir=download_dir,
            served_model_name=served_model_name,
        )

    def generate(self, prompts: Sequence[str], config: GenerationConfig | None = None) -> list[str]:
        from vllm import SamplingParams

        cfg = config or GenerationConfig()
        params = SamplingParams(
            max_tokens=cfg.max_tokens,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
        )
        lora_request = None
        if self.lora_path:
            from vllm.lora.request import LoRARequest

            lora_request = LoRARequest(self.lora_name, 1, self.lora_path)
        outputs = self.llm.generate(list(prompts), params, lora_request=lora_request)
        return [out.outputs[0].text.strip() for out in outputs]


def resolve_model_path(model: str) -> str:
    candidate = Path(model).expanduser()
    if candidate.exists():
        return str(candidate.resolve())
    return model

from __future__ import annotations

import asyncio
import json
import math
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def cosine_similarity(left: list[float], right: list[float]) -> float:
    """Return a finite cosine similarity for two non-empty vectors."""

    if not left or not right:
        raise ValueError("semantic similarity vectors must be non-empty.")
    if len(left) != len(right):
        raise ValueError("semantic similarity vectors must have equal length.")
    if any(not math.isfinite(value) for value in (*left, *right)):
        raise ValueError("semantic similarity vectors must contain finite values.")

    dot = math.fsum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(math.fsum(value * value for value in left))
    right_norm = math.sqrt(math.fsum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        raise ValueError("semantic similarity vectors must have non-zero norm.")
    return max(-1.0, min(1.0, dot / (left_norm * right_norm)))


@dataclass(slots=True)
class _PendingEmbeddingRequest:
    texts: list[str]
    future: asyncio.Future[list[list[float]]]


class LocalEmbeddingCaller:
    """Dynamically batch text on one process-local Hugging Face model."""

    def __init__(
        self,
        *,
        model_path: str,
        device: str = "cuda",
        dtype: str = "bfloat16",
        max_length: int = 8192,
        batch_wait_ms: float = 2.0,
        max_batch_texts: int = 64,
        max_batch_tokens: int = 32768,
    ) -> None:
        path = Path(model_path).expanduser()
        if not path.is_dir():
            raise ValueError(f"Local embedding model directory does not exist: {path}")
        if not str(device).strip():
            raise ValueError("Local embedding device must be non-empty.")
        normalized_dtype = str(dtype).strip().lower()
        if normalized_dtype not in {"float32", "float16", "bfloat16"}:
            raise ValueError(
                "Local embedding dtype must be float32, float16, or bfloat16."
            )
        if int(max_length) <= 0:
            raise ValueError("Local embedding max_length must be positive.")
        if float(batch_wait_ms) < 0.0:
            raise ValueError("Local embedding batch_wait_ms must be non-negative.")
        if int(max_batch_texts) <= 0:
            raise ValueError("Local embedding max_batch_texts must be positive.")
        if int(max_batch_tokens) <= 0:
            raise ValueError("Local embedding max_batch_tokens must be positive.")
        self.model_path = str(path.resolve())
        self.device = str(device).strip()
        self.dtype = normalized_dtype
        self.max_length = int(max_length)
        self.batch_wait_ms = float(batch_wait_ms)
        self.max_batch_texts = int(max_batch_texts)
        self.max_batch_tokens = int(max_batch_tokens)
        self.pooling_mode = self._read_pooling_mode(path)
        self._tokenizer: Any | None = None
        self._model: Any | None = None
        self._pending_requests: list[_PendingEmbeddingRequest] = []
        self._batch_task: asyncio.Task[None] | None = None
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="tutor-embedding",
        )

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts or any(not text.strip() for text in texts):
            raise ValueError("embedding inputs must be non-empty strings.")
        loop = asyncio.get_running_loop()
        if (
            self._batch_task is not None
            and not self._batch_task.done()
            and self._batch_task.get_loop() is not loop
        ):
            raise RuntimeError(
                "One LocalEmbeddingCaller cannot serve multiple event loops concurrently."
            )
        future = loop.create_future()
        self._pending_requests.append(
            _PendingEmbeddingRequest(texts=list(texts), future=future)
        )
        if self._batch_task is None or self._batch_task.done():
            self._batch_task = loop.create_task(self._flush_pending_requests())
        return await future

    async def _flush_pending_requests(self) -> None:
        await asyncio.sleep(self.batch_wait_ms / 1000.0)
        requests = self._pending_requests
        self._pending_requests = []
        texts = [text for request in requests for text in request.texts]
        loop = asyncio.get_running_loop()
        try:
            vectors = await loop.run_in_executor(
                self._executor, self._embed_sync, texts
            )
            if len(vectors) != len(texts):
                raise RuntimeError(
                    "Embedding output count does not match the input text count."
                )
            offset = 0
            for request in requests:
                next_offset = offset + len(request.texts)
                if not request.future.cancelled():
                    request.future.set_result(vectors[offset:next_offset])
                offset = next_offset
        except Exception as exc:
            for request in requests:
                if not request.future.cancelled():
                    request.future.set_exception(exc)
        finally:
            self._batch_task = None
            if self._pending_requests:
                self._batch_task = loop.create_task(self._flush_pending_requests())

    @staticmethod
    def _read_pooling_mode(model_path: Path) -> str:
        pooling_config_path = model_path / "1_Pooling" / "config.json"
        if not pooling_config_path.is_file():
            return "mean"
        with pooling_config_path.open(encoding="utf-8") as handle:
            config = json.load(handle)
        if config.get("pooling_mode_cls_token"):
            return "cls"
        if config.get("pooling_mode_mean_tokens"):
            return "mean"
        raise ValueError(
            "Local embedding model must use CLS or mean-token pooling: "
            f"{pooling_config_path}"
        )

    def _load_model(self) -> tuple[Any, Any]:
        import torch
        from transformers import AutoModel, AutoTokenizer

        dtype = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }[self.dtype]

        tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            local_files_only=True,
        )
        model = AutoModel.from_pretrained(
            self.model_path,
            local_files_only=True,
            dtype=dtype,
        )
        model.to(self.device)
        model.eval()
        return tokenizer, model

    def _embed_sync(self, texts: list[str]) -> list[list[float]]:
        import torch

        # A dedicated single-thread executor keeps model forward off the rollout
        # event loop and serializes concurrent episode requests without occupying
        # the event loop's shared thread pool.
        if self._tokenizer is None or self._model is None:
            self._tokenizer, self._model = self._load_model()
        encoded = self._tokenizer(
            texts,
            padding=False,
            truncation=True,
            max_length=self.max_length,
        )
        lengths = [len(input_ids) for input_ids in encoded["input_ids"]]
        batches = self._allocate_microbatches(lengths)
        vectors: list[list[float] | None] = [None] * len(texts)
        for indices in batches:
            inputs = self._tokenizer(
                [texts[index] for index in indices],
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            inputs = {name: tensor.to(self.device) for name, tensor in inputs.items()}
            with torch.inference_mode():
                hidden = self._model(**inputs).last_hidden_state

            if self.pooling_mode == "cls":
                pooled = hidden[:, 0]
            else:
                attention_mask = inputs["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * attention_mask).sum(dim=1)
                pooled = pooled / attention_mask.sum(dim=1).clamp_min(1e-9)
            normalized = torch.nn.functional.normalize(pooled, p=2, dim=1)
            for index, vector in zip(
                indices,
                normalized.float().cpu().tolist(),
                strict=True,
            ):
                vectors[index] = vector
        if any(vector is None for vector in vectors):
            raise RuntimeError("Embedding microbatching did not produce every vector.")
        return [vector for vector in vectors if vector is not None]

    def _allocate_microbatches(self, lengths: list[int]) -> list[list[int]]:
        """Bucket similar lengths while bounding padded tokens per GPU forward."""

        batches: list[list[int]] = []
        current: list[int] = []
        current_max_length = 0
        for index in sorted(range(len(lengths)), key=lengths.__getitem__):
            next_max_length = max(current_max_length, lengths[index])
            exceeds_texts = len(current) >= self.max_batch_texts
            exceeds_tokens = (
                next_max_length * (len(current) + 1) > self.max_batch_tokens
            )
            if current and (exceeds_texts or exceeds_tokens):
                batches.append(current)
                current = []
                current_max_length = 0
            current.append(index)
            current_max_length = max(current_max_length, lengths[index])
        if current:
            batches.append(current)
        return batches


_LOCAL_CALLERS: dict[
    tuple[str, str, str, int, float, int, int], LocalEmbeddingCaller
] = {}
_LOCAL_CALLERS_LOCK = threading.Lock()


def get_local_embedding_caller(
    *,
    model_path: str,
    device: str = "cuda",
    dtype: str = "bfloat16",
    max_length: int = 8192,
    batch_wait_ms: float = 2.0,
    max_batch_texts: int = 64,
    max_batch_tokens: int = 32768,
) -> LocalEmbeddingCaller:
    """Return one shared lazy embedding caller per process and configuration."""

    resolved_path = str(Path(model_path).expanduser().resolve())
    key = (
        resolved_path,
        str(device).strip(),
        str(dtype).strip().lower(),
        int(max_length),
        float(batch_wait_ms),
        int(max_batch_texts),
        int(max_batch_tokens),
    )
    with _LOCAL_CALLERS_LOCK:
        caller = _LOCAL_CALLERS.get(key)
        if caller is None:
            caller = LocalEmbeddingCaller(
                model_path=resolved_path,
                device=device,
                dtype=dtype,
                max_length=max_length,
                batch_wait_ms=batch_wait_ms,
                max_batch_texts=max_batch_texts,
                max_batch_tokens=max_batch_tokens,
            )
            _LOCAL_CALLERS[key] = caller
        return caller

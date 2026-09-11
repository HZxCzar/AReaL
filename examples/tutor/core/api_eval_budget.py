"""Single-process, usage-based spending guard (not a provider billing cap)."""

import json
import math
import os
from pathlib import Path


class TeacherBudget:
    """Account for returned tokens, including cached input and native reasoning.

    Rates are USD per million tokens: input, cache read, cache write, output.
    Concurrent requests already sent may exceed the threshold. Missing usage or
    a failed request stops further work unless an explicit estimated reservation
    is configured via TUTOR_API_UNKNOWN_REQUEST_RESERVE_USD. Reservations are
    estimates, not bounds on provider charges, and count toward the same limit.
    """

    def __init__(self, path: Path, limit: float, rates: list[float]):
        if not math.isfinite(limit) or limit <= 0:
            raise ValueError("Budget must be finite and positive")
        if len(rates) != 4 or any(not math.isfinite(x) or x < 0 for x in rates):
            raise ValueError("Supply four finite non-negative token prices")
        self.path = path
        self.limit = limit
        self.rates = rates
        self.spent = 0.0
        self.unknown = False
        self.unknown_count = 0
        self.unknown_reserve = float(
            os.getenv("TUTOR_API_UNKNOWN_REQUEST_RESERVE_USD", "0")
        )
        if not math.isfinite(self.unknown_reserve) or self.unknown_reserve < 0:
            raise ValueError("Unknown-request reserve must be finite and non-negative")
        status_path = path.with_name("budget_status.json")
        if self.unknown_reserve and status_path.exists():
            previous = json.loads(status_path.read_text())
            self.unknown_reserve = max(
                self.unknown_reserve, previous.get("unknown_request_reserve_usd", 0)
            )
        if path.exists():
            for line in path.read_text().splitlines():
                self._account(json.loads(line).get("usage"))

    def _account(self, usage):
        if not usage or not {"prompt_tokens", "completion_tokens"} <= usage.keys():
            self.unknown = True
            self.unknown_count += 1
            return
        details = usage.get("prompt_tokens_details") or {}
        read = details.get("cached_tokens", 0) or 0
        write = details.get("cache_write_tokens", 0) or 0
        ordinary = max(0, usage["prompt_tokens"] - read - write)
        completion = usage["completion_tokens"]
        total = usage.get("total_tokens")
        if total is not None:
            if not math.isfinite(total) or total < 0:
                self.unknown = True
                self.unknown_count += 1
                return
            # Gemini's compatible API can omit thinking from completion_tokens
            # while including it in total_tokens. OpenAI already includes it.
            # Taking the larger count covers both without double charging.
            completion = max(completion, total - usage["prompt_tokens"])
        counts = [ordinary, read, write, usage["completion_tokens"], completion]
        if any(not math.isfinite(x) or x < 0 for x in counts):
            self.unknown = True
            self.unknown_count += 1
            return
        self.spent += (
            sum(
                n * rate
                for n, rate in zip([ordinary, read, write, completion], self.rates)
            )
            / 1e6
        )

    def stopped(self):
        return (self.unknown and not self.unknown_reserve) or (
            self.spent + self.unknown_count * self.unknown_reserve >= self.limit
        )

    def check(self):
        if self.stopped():
            raise RuntimeError("Teacher budget stopped; see budget_status.json")

    def record(self, record):
        with self.path.open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        self._account(record.get("usage"))
        self.save_status()

    def save_status(self):
        status = self.path.with_name("budget_status.json")
        temporary = status.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(
                {
                    "limit_usd": self.limit,
                    "estimated_spend_usd": self.spent,
                    "rates_per_million": self.rates,
                    "unknown_usage": self.unknown,
                    "unknown_request_count": self.unknown_count,
                    "unknown_request_reserve_usd": self.unknown_reserve,
                    "reserved_unknown_usd": self.unknown_count * self.unknown_reserve,
                    "budget_used_usd": self.spent
                    + self.unknown_count * self.unknown_reserve,
                    "stopped": self.stopped(),
                    "scope": "This output directory only; not a provider billing cap",
                },
                indent=2,
            )
            + "\n"
        )
        temporary.replace(status)

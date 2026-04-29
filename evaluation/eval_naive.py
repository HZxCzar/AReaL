#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluate a local post-trained Qwen/Qwen3-style model on a math dataset.

Expected dataset format (jsonl), one object per line.

Math500-style example:
{
  "problem": "...",
  "solution": "...",
  "answer": "\\left( 3, \\frac{\\pi}{2} \\right)",
  "subject": "Precalculus",
  "level": 2,
  "unique_id": "test/precalculus/807.json"
}

AIME-style example:
{
  "problem": "...",
  "answer": 610,
  "id": "22"
}

AMC-style example:
{
  "id": 0,
  "problem": "...",
  "question": "...",
  "answer": 27.0
}

Example:
python eval_math500_vllm.py \
    --model /path/to/your/local/model \
    --data data/math_500/test.jsonl \
    --template qwen3-think \
    --tensor-parallel-size 1 \
    --batch-size 64 \
    --max-tokens 4096 \
    --output-dir outputs/qwen3_math500

Notes:
- The script first tries to compare extracted boxed answers after normalization.
- If sympy is available, it will also try a symbolic equivalence fallback.
- For Qwen3 "think" style prompts, we directly control the raw prompt text.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from vllm import LLM, SamplingParams


# =========================
# Prompt templates
# =========================

PROMPT_TEMPLATES = {
    "qwen25-math-cot": (
        "<|im_start|>system\nPlease reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>\n"
        "<|im_start|>user\n{input}<|im_end|>\n"
        "<|im_start|>assistant\n",
        "{output}",
        "\n\n",
    ),
    "qwen3-think": (
        "<|im_start|>user\n{input}\nPlease reason step by step, and put your final answer within \\boxed{{}}./think<|im_end|>\n"
        "<|im_start|>assistant\n<think>",
        "{output}",
        "\n\n",
    ),
    "qwen3": (
        "<|im_start|>user\n{input}\nPlease reason step by step, and put your final answer within \\boxed{{}}./no_think<|im_end|>\n"
        "<|im_start|>assistant\n<think></think>",
        "{output}",
        "\n\n",
    ),
    "qwen3-think-pure": (
        "<|im_start|>user\n{input}\n/think<|im_end|>\n"
        "<|im_start|>assistant\n<think>",
        "{output}",
        "\n\n",
    ),
}


# =========================
# Utilities
# =========================

def load_jsonl(path: str) -> List[Dict[str, Any]]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line_id, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_id} of {path}: {e}") from e
    return data


def save_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def chunked(xs: List[Any], n: int) -> List[List[Any]]:
    return [xs[i:i + n] for i in range(0, len(xs), n)]


def render_prompt(template_name: str, problem_text: str) -> str:
    if template_name not in PROMPT_TEMPLATES:
        raise KeyError(f"Unknown template: {template_name}")
    prompt_prefix, _, _ = PROMPT_TEMPLATES[template_name]
    return prompt_prefix.format(input=problem_text)


def strip_outer_dollars(s: str) -> str:
    s = s.strip()
    if len(s) >= 2 and s[0] == "$" and s[-1] == "$":
        return s[1:-1].strip()
    return s


def remove_latex_wrappers(s: str) -> str:
    s = s.strip()
    # Remove \left and \right
    s = s.replace("\\left", "").replace("\\right", "")
    # Remove \!
    s = s.replace("\\!", "")
    # Normalize whitespace
    s = " ".join(s.split()).strip()
    return s


def normalize_latex_answer(s: Optional[Any]) -> str:
    if s is None:
        return ""

    s = str(s)
    s = strip_outer_dollars(s)
    s = remove_latex_wrappers(s)

    # Common cleanup
    s = s.replace("{ ", "{").replace(" }", "}")
    s = s.replace("( ", "(").replace(" )", ")")
    s = s.replace("[ ", "[").replace(" ]", "]")
    s = s.replace(" ,", ",")
    s = s.strip(" .\n\t")

    # Remove surrounding \boxed{...} if present
    boxed = extract_last_boxed(s)
    if boxed is not None:
        s = boxed.strip()

    # Normalize equivalent TeX aliases a bit
    s = s.replace("\\dfrac", "\\frac")
    s = s.replace("\\tfrac", "\\frac")
    s = "".join(s.split())

    return s


def extract_last_boxed(text: str) -> Optional[str]:
    """
    Extract the content of the last \\boxed{...} occurrence, handling nested braces.
    Returns the inner content without \\boxed{ }.
    """
    if not text:
        return None

    marker = "\\boxed{"
    start = text.rfind(marker)
    if start == -1:
        return None

    i = start + len(marker)
    depth = 1
    out_chars = []

    while i < len(text):
        ch = text[i]
        if ch == "{":
            depth += 1
            out_chars.append(ch)
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return "".join(out_chars).strip()
            out_chars.append(ch)
        else:
            out_chars.append(ch)
        i += 1

    return None


def extract_answer_fallback(text: str) -> str:
    """
    Fallback if no boxed answer is found.
    Heuristics:
    - take the last non-empty line
    - remove closing special tokens
    """
    if not text:
        return ""

    text = text.replace("<|im_end|>", "").strip()
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return text.strip()
    return lines[-1]


def extract_predicted_answer(raw_text: str) -> str:
    boxed = extract_last_boxed(raw_text)
    if boxed is not None:
        return boxed.strip()
    return extract_answer_fallback(raw_text)


def parse_balanced_group(text: str, start: int, open_ch: str, close_ch: str) -> Tuple[Optional[str], int]:
    if start >= len(text) or text[start] != open_ch:
        return None, start

    depth = 1
    i = start + 1
    chars: List[str] = []
    while i < len(text):
        ch = text[i]
        if ch == open_ch:
            depth += 1
            chars.append(ch)
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                return "".join(chars), i + 1
            chars.append(ch)
        else:
            chars.append(ch)
        i += 1
    return None, start


def replace_frac_calls(text: str, token: str, open_ch: str, close_ch: str) -> str:
    if not text or token not in text:
        return text

    out: List[str] = []
    i = 0
    token_len = len(token)
    while i < len(text):
        if text.startswith(token, i):
            num, next_i = parse_balanced_group(text, i + token_len, open_ch, close_ch)
            if num is None:
                out.append(text[i])
                i += 1
                continue
            den, end_i = parse_balanced_group(text, next_i, open_ch, close_ch)
            if den is None:
                out.append(text[i])
                i += 1
                continue

            out.append(f"(({replace_frac_calls(num, token, open_ch, close_ch)})/({replace_frac_calls(den, token, open_ch, close_ch)}))")
            i = end_i
            continue

        out.append(text[i])
        i += 1

    return "".join(out)


def preprocess_math_expr(expr: str, frac_open: str, frac_close: str) -> str:
    expr = expr.strip()
    expr = expr.replace("\\left", "").replace("\\right", "")
    expr = expr.replace("\\!", "")
    expr = expr.replace("\\cdot", "*")
    expr = expr.replace("^", "**")
    expr = expr.replace("\\dfrac", "\\frac")
    expr = expr.replace("\\tfrac", "\\frac")
    expr = expr.replace("\\pi", "pi")
    expr = replace_frac_calls(expr, "\\frac", frac_open, frac_close)
    return expr


def safe_ast_eval(expr: str) -> Optional[float]:
    allowed_binops = {
        ast.Add: lambda a, b: a + b,
        ast.Sub: lambda a, b: a - b,
        ast.Mult: lambda a, b: a * b,
        ast.Div: lambda a, b: a / b,
        ast.Pow: lambda a, b: a ** b,
    }
    allowed_unaryops = {
        ast.UAdd: lambda a: a,
        ast.USub: lambda a: -a,
    }

    def eval_node(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return eval_node(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.Name) and node.id == "pi":
            return float(math.pi)
        if isinstance(node, ast.BinOp) and type(node.op) in allowed_binops:
            return allowed_binops[type(node.op)](eval_node(node.left), eval_node(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in allowed_unaryops:
            return allowed_unaryops[type(node.op)](eval_node(node.operand))
        raise ValueError(f"Unsupported AST node: {type(node).__name__}")

    try:
        tree = ast.parse(expr, mode="eval")
        return float(eval_node(tree))
    except Exception:
        return None


def safe_float_eval(expr: str) -> Optional[float]:
    try:
        expr = preprocess_math_expr(expr, "{", "}")
        return safe_ast_eval(expr)
    except Exception:
        return None


def try_sympy_equiv(a: str, b: str) -> bool:
    """
    Best-effort symbolic equivalence.
    Returns False if sympy is unavailable or parsing fails.
    """
    try:
        import sympy as sp
        from sympy.parsing.sympy_parser import parse_expr
    except Exception:
        return False

    def preprocess(x: str) -> str:
        x = preprocess_math_expr(x, "{", "}")
        return x.replace("{", "(").replace("}", ")")

    try:
        aa = preprocess(a)
        bb = preprocess(b)

        # Tuple/list answers: compare elementwise
        if aa.startswith("(") and aa.endswith(")") and bb.startswith("(") and bb.endswith(")"):
            a_items = split_top_level_csv(aa[1:-1])
            b_items = split_top_level_csv(bb[1:-1])
            if len(a_items) != len(b_items):
                return False
            for x, y in zip(a_items, b_items):
                ex = parse_expr(x, local_dict={"pi": sp.pi})
                ey = parse_expr(y, local_dict={"pi": sp.pi})
                if sp.simplify(ex - ey) != 0:
                    return False
            return True

        ex = parse_expr(aa, local_dict={"pi": sp.pi})
        ey = parse_expr(bb, local_dict={"pi": sp.pi})
        return sp.simplify(ex - ey) == 0
    except Exception:
        return False


def split_top_level_csv(s: str) -> List[str]:
    parts = []
    cur = []
    depth = 0
    for ch in s:
        if ch in "([{":
            depth += 1
            cur.append(ch)
        elif ch in ")]}":
            depth -= 1
            cur.append(ch)
        elif ch == "," and depth == 0:
            parts.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    if cur:
        parts.append("".join(cur).strip())
    return parts


def answers_equal(pred: str, gold: str) -> Tuple[bool, str]:
    """
    Returns:
      (is_correct, match_type)
    """
    pred_norm = normalize_latex_answer(pred)
    gold_norm = normalize_latex_answer(gold)

    if pred_norm == gold_norm:
        return True, "normalized_exact"

    # Numeric fallback
    pred_num = safe_float_eval(pred_norm)
    gold_num = safe_float_eval(gold_norm)
    if pred_num is not None and gold_num is not None:
        if abs(pred_num - gold_num) <= 1e-8:
            return True, "numeric"

    # Symbolic fallback
    if try_sympy_equiv(pred_norm, gold_norm):
        return True, "sympy"

    return False, "none"


def get_example_id(example: Dict[str, Any]) -> str:
    return str(example.get("unique_id") or example.get("id") or "")


def get_gold_answer(example: Dict[str, Any]) -> str:
    return normalize_latex_answer(example.get("answer"))


def get_problem_text(example: Dict[str, Any]) -> str:
    problem = example.get("problem")
    if problem is None or str(problem).strip() == "":
        problem = example.get("question", "")
    return str(problem)


@dataclass
class EvalItem:
    unique_id: str
    subject: Optional[str]
    level: Optional[int]
    problem: str
    gold_answer: str
    raw_generation: str
    pred_answer: str
    correct: bool
    match_type: str


# =========================
# Main evaluation
# =========================

def build_llm(args: argparse.Namespace) -> LLM:
    return LLM(
        model=args.model,
        tokenizer=args.tokenizer or args.model,
        trust_remote_code=args.trust_remote_code,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype=args.dtype,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        swap_space=args.swap_space,
        enable_prefix_caching=args.enable_prefix_caching,
    )


def build_sampling_params(args: argparse.Namespace) -> SamplingParams:
    stop = ["<|im_end|>"]
    if args.extra_stop:
        stop.extend(args.extra_stop)

    return SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
        n=1,
        stop=stop,
        repetition_penalty=args.repetition_penalty,
        skip_special_tokens=False,
    )


def evaluate(args: argparse.Namespace) -> Dict[str, Any]:
    os.makedirs(args.output_dir, exist_ok=True)

    data = load_jsonl(args.data)
    if args.max_samples is not None:
        data = data[: args.max_samples]

    llm = build_llm(args)
    sampling_params = build_sampling_params(args)

    prompts = [render_prompt(args.template, get_problem_text(ex)) for ex in data]

    results: List[EvalItem] = []

    for batch_indices in chunked(list(range(len(data))), args.batch_size):
        batch_prompts = [prompts[i] for i in batch_indices]
        batch_data = [data[i] for i in batch_indices]

        outputs = llm.generate(batch_prompts, sampling_params, use_tqdm=True)

        for ex, out in zip(batch_data, outputs):
            # vLLM returns a list of candidate outputs in out.outputs
            raw_text = out.outputs[0].text if out.outputs else ""
            pred_answer = extract_predicted_answer(raw_text)
            gold_answer = get_gold_answer(ex)
            correct, match_type = answers_equal(pred_answer, gold_answer)

            results.append(
                EvalItem(
                    unique_id=get_example_id(ex),
                    subject=ex.get("subject"),
                    level=ex.get("level"),
                    problem=get_problem_text(ex),
                    gold_answer=gold_answer,
                    raw_generation=raw_text,
                    pred_answer=pred_answer,
                    correct=correct,
                    match_type=match_type,
                )
            )

    total = len(results)
    num_correct = sum(int(x.correct) for x in results)
    accuracy = num_correct / total if total > 0 else 0.0

    by_subject: Dict[str, Dict[str, Any]] = {}
    by_level: Dict[str, Dict[str, Any]] = {}

    for item in results:
        subj = item.subject if item.subject is not None else "UNKNOWN"
        lvl = str(item.level) if item.level is not None else "UNKNOWN"

        if subj not in by_subject:
            by_subject[subj] = {"total": 0, "correct": 0}
        by_subject[subj]["total"] += 1
        by_subject[subj]["correct"] += int(item.correct)

        if lvl not in by_level:
            by_level[lvl] = {"total": 0, "correct": 0}
        by_level[lvl]["total"] += 1
        by_level[lvl]["correct"] += int(item.correct)

    for d in (by_subject, by_level):
        for k, v in d.items():
            v["accuracy"] = v["correct"] / v["total"] if v["total"] > 0 else 0.0

    result_rows = [
        {
            "unique_id": x.unique_id,
            "subject": x.subject,
            "level": x.level,
            "problem": x.problem,
            "gold_answer": x.gold_answer,
            "raw_generation": x.raw_generation,
            "pred_answer": x.pred_answer,
            "correct": x.correct,
            "match_type": x.match_type,
        }
        for x in results
    ]

    save_jsonl(os.path.join(args.output_dir, "predictions.jsonl"), result_rows)

    summary = {
        "model": args.model,
        "tokenizer": args.tokenizer or args.model,
        "data": args.data,
        "template": args.template,
        "num_samples": total,
        "num_correct": num_correct,
        "accuracy": accuracy,
        "by_subject": by_subject,
        "by_level": by_level,
        "sampling": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "max_tokens": args.max_tokens,
            "repetition_penalty": args.repetition_penalty,
        },
    }

    with open(os.path.join(args.output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a local Qwen/Qwen3 model on a math jsonl dataset using vLLM.")

    # Core paths
    parser.add_argument("--model", type=str, required=True, help="Local model path or HF identifier.")
    parser.add_argument("--tokenizer", type=str, default=None, help="Optional tokenizer path. Defaults to model.")
    parser.add_argument("--data", type=str, default="data/math_500/test.jsonl", help="Path to test jsonl.")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save predictions and summary.")

    # Prompting
    parser.add_argument(
        "--template",
        type=str,
        default="qwen3-think",
        choices=sorted(PROMPT_TEMPLATES.keys()),
        help="Prompt template name.",
    )

    # vLLM/model settings
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--dtype", type=str, default="auto", help="auto, float16, bfloat16, etc.")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--swap-space", type=int, default=4)
    parser.add_argument("--enable-prefix-caching", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")

    # Generation
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--extra-stop", type=str, nargs="*", default=[])

    # Debug / slicing
    parser.add_argument("--max-samples", type=int, default=None, help="Only evaluate the first N samples.")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluate(args)


if __name__ == "__main__":
    print("Starting Math 500 Evaluation...")
    main()

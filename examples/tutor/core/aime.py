from __future__ import annotations

import json
import re

from .text import strip_reasoning_for_context
from .types import JudgeResult


def score_aime_answer(task: str, ground_truth: str, student_answer: str) -> JudgeResult:
    extracted_answer = official_extract_aime_answer(student_answer)
    normalized_prediction = (
        official_strip_string(extracted_answer) if extracted_answer else ""
    )
    normalized_target = official_strip_string(ground_truth)
    correct = official_is_equiv(extracted_answer, ground_truth)
    raw_result = {
        "method": "lm_eval_aime_exact_match",
        "task": task,
        "student_answer": strip_reasoning_for_context(student_answer),
        "extracted_answer": extracted_answer,
        "normalized_prediction": normalized_prediction,
        "normalized_target": normalized_target,
    }
    return JudgeResult(
        raw_output=json.dumps(
            {
                "correct": correct,
                "feedback": "Correct." if correct else "Incorrect.",
                "scoring": raw_result,
            },
            ensure_ascii=True,
            indent=2,
        ),
        correct=correct,
        feedback="Correct." if correct else "Incorrect.",
        parse_error=None,
        raw_result=raw_result,
    )


def _fix_fracs(string: str) -> str:
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if len(substr) > 0 and substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except AssertionError:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    return new_str


def _fix_a_slash_b(string: str) -> str:
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a = int(a)
        b = int(b)
        assert string == f"{a}/{b}"
        return "\\frac{" + str(a) + "}{" + str(b) + "}"
    except Exception:
        return string


def _remove_right_units(string: str) -> str:
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        return splits[0]
    return string


def _fix_sqrt(string: str) -> str:
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if not split:
            new_string += "\\sqrt"
            continue
        if split[0] != "{":
            a = split[0]
            new_string += "\\sqrt{" + a + "}" + split[1:]
        else:
            new_string += "\\sqrt" + split
    return new_string


def _strip_string(string: str) -> str:
    string = string.replace("\n", "")
    string = string.replace("\\!", "")
    string = string.replace("\\\\", "\\")
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")
    string = string.replace("\\$", "")
    string = _remove_right_units(string)
    string = string.replace("\\%", "")
    string = string.replace("\\%", "")
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]
    string = _fix_sqrt(string)
    string = string.replace(" ", "")
    string = _fix_fracs(string)
    if string == "0.5":
        string = "\\frac{1}{2}"
    string = _fix_a_slash_b(string)
    return string


def official_strip_string(string: str) -> str:
    return _strip_string(string)


def official_is_equiv(prediction: str, reference: str) -> bool:
    return official_strip_string(prediction) == official_strip_string(reference)


def official_extract_aime_answer(response: str) -> str:
    matches = list(
        re.finditer(r"(?:^|[^0-9])([0-9]{1,4})(?:[^0-9]|$)", response or "")
    )
    if not matches:
        return (response or "").strip()
    return matches[-1].group(1)

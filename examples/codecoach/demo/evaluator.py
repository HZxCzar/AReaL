import importlib.util
import json
import sys
from pathlib import Path


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("candidate_module", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def main():
    code_path = Path(sys.argv[1]).resolve()
    result_path = Path(sys.argv[2]).resolve()
    result = {"score": 0.0, "target_ratio": 0.0, "validity": 0.0, "success": False}
    try:
        module = load_module(code_path)
        answer = module.construct_answer()
        score = 1.0 if answer == 42 else 0.0
        result.update(
            {
                "score": score,
                "target_ratio": score,
                "validity": 1.0,
                "success": bool(score >= 1.0),
                "answer": answer,
            }
        )
    except Exception as exc:
        result["error"] = str(exc)
    result_path.write_text(json.dumps(result, ensure_ascii=True, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

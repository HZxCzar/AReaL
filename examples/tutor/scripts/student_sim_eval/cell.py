"""Reuse API-run manifests, budget accounting, preflight, logs and resume unchanged."""

from examples.tutor.scripts.api_run import runner


def main(evaluator_module="examples.tutor.scripts.student_sim_eval.evaluate"):
    original = runner.prepare

    def prepare(*args, **kwargs):
        command, manifest = original(*args, **kwargs)
        assert command[2] == "examples.tutor.evaluate_teacher_api"
        command[2] = evaluator_module
        return command, manifest

    try:
        runner.prepare = prepare
        runner.main()
    finally:
        runner.prepare = original


if __name__ == "__main__":
    main()

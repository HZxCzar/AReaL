"""Standard API-run accounting and resume, with the strict student evaluator."""

from examples.tutor.scripts.student_sim_eval import cell as shared


def main():
    shared.main("examples.tutor.scripts.student_sim_eval.prompted_strict.evaluate")


if __name__ == "__main__":
    main()

# MathTutorBench Comparison

All metrics are reported on a 0–1 scale and higher is better. `Pedagogy Average`
is the arithmetic mean of the four Ped-RM win rates and is not an official
MathTutorBench metric.

| Source | Model | Problem Solving (Accuracy) | Socratic Questioning (BLEU) | Solution Correctness (F1) | Mistake Location (Micro-F1) | Mistake Correction (Accuracy) | Scaffolding Generation (Ped-RM Win Rate) | Pedagogical Instruction Following (Ped-RM Win Rate) | Scaffolding Generation [Hard] (Ped-RM Win Rate) | Pedagogical Instruction Following [Hard] (Ped-RM Win Rate) | Pedagogy Average |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Current MathTutorBench leaderboard | `Qwen/Qwen2.5-7B-Instruct` | 0.85 | 0.19 | 0.64 | 0.49 | 0.52 | 0.44 | 0.40 | 0.42 | 0.44 | 0.425 |
| Current MathTutorBench leaderboard | `eth-nlped/TutorRL-7B` | 0.77 | 0.23 | 0.65 | 0.36 | 0.75 | 0.53 | 0.70 | 0.55 | 0.66 | 0.610 |
| Local evaluation | `Qwen/Qwen3-8B` | 0.921 | 0.253 | 0.685 | 0.371 | 0.675 | 0.331 | 0.740 | 0.324 | 0.706 | 0.525 |
| Local evaluation | `0906-pedagogical-rl-qwen3-8b-lr5e-5-8gpu@globalstep999` | 0.927 | 0.245 | 0.688 | 0.330 | 0.711 | 0.497 | 0.738 | 0.373 | 0.774 | 0.596 |
| Local evaluation | `20260901_182144_0901-preference-v3-reward-v3-none-8gpu@globalstep999` | 0.926 | 0.232 | 0.691 | 0.262 | 0.635 | 0.670 | 0.811 | 0.633 | 0.780 | 0.724 |
| Local evaluation | `20260901_182029_0901-preference-v3-reward-v3-all-id-8gpu@globalstep999` | 0.929 | 0.227 | 0.690 | 0.289 | 0.692 | 0.661 | 0.843 | 0.606 | 0.829 | 0.734 |
| Local evaluation | `20260901_182029_0901-preference-v3-reward-v3-all-id-8gpu@globalstep1499` | 0.932 | 0.219 | 0.688 | 0.303 | 0.668 | 0.721 | 0.903 | 0.667 | 0.872 | 0.791 |
| Local evaluation | `20260908_0901-reward-v4-all-id-fork775-8gpu@globalstep999` | 0.935 | 0.198 | 0.691 | 0.372 | 0.346 | 0.438 | 0.303 | 0.330 | 0.434 | 0.376 |

## Within-base changes

| Method | Base Pedagogy Average | Trained Pedagogy Average | Absolute Change |
|---|---:|---:|---:|
| `Qwen/Qwen2.5-7B-Instruct` → `eth-nlped/TutorRL-7B` (current MathTutorBench leaderboard) | 0.425 | 0.610 | +0.185 |
| `Qwen/Qwen3-8B` → `PedagogicalRL@globalstep999` (local evaluation) | 0.525 | 0.596 | +0.070 |
| `Qwen/Qwen3-8B` → `reward-v3-none@globalstep999` (local evaluation) | 0.525 | 0.724 | +0.199 |
| `Qwen/Qwen3-8B` → `20260901_182029_0901-preference-v3-reward-v3-all-id-8gpu@globalstep999` (local evaluation) | 0.525 | 0.734 | +0.209 |
| `Qwen/Qwen3-8B` → `20260901_182029_0901-preference-v3-reward-v3-all-id-8gpu@globalstep1499` (local evaluation; 1,500 steps) | 0.525 | 0.791 | +0.265 |
| `Qwen/Qwen3-8B` → `20260908_0901-reward-v4-all-id-fork775-8gpu@globalstep999` (local evaluation) | 0.525 | 0.376 | -0.149 |

## Sources and comparability

- [Current MathTutorBench leaderboard](https://github.com/eth-lre/mathtutorbench)
- [Local `Qwen/Qwen3-8B` results](results/Qwen3-8B-base/summary.yaml)
- [Local `PedagogicalRL@globalstep999` results](results/0906-pedagogical-rl-qwen3-8b-lr5e-5-8gpu/epoch21epochstep12globalstep999/summary.yaml)
- [Local `reward-v3-none@globalstep999` results](results/20260901_182144_0901-preference-v3-reward-v3-none-8gpu/epoch21epochstep12globalstep999/summary.yaml)
- [Local trained-checkpoint results](results/20260901_182029_0901-preference-v3-reward-v3-all-id-8gpu/epoch21epochstep12globalstep999/summary.yaml)
- [Local `reward-v3-all-id@globalstep1499` results (1,500 steps)](results/20260901_182029_0901-preference-v3-reward-v3-all-id-8gpu/epoch31epochstep42globalstep1499/summary.yaml)
- [Local `reward-v4-all-id-fork775@globalstep999` results](results/20260908_0901-reward-v4-all-id-fork775-8gpu/epoch21epochstep12globalstep999/summary.yaml)

The two Qwen2.5 rows use the current official MathTutorBench leaderboard values.
The six Qwen3 rows were evaluated with the same local runner. PedagogicalRL and
our methods were trained from the same `Qwen/Qwen3-8B` base model, while retaining
each method's native training protocol. The `globalstep999` checkpoints share
a 1,000-step training budget; the `reward-v3-all-id@globalstep1499` checkpoint
has 1,500 completed updates and is not a matched-budget comparison.

The 1,500-step evaluation is complete for all nine tasks and all four Ped-RM
scores, using temperature 0, seed 42, and a 2,048-token output limit. Its
Pedagogy Average is 0.791, compared with 0.734 at 1,000 steps.

The PedagogicalRL row uses the September 8 evaluation of the new run with LoRA
learning rate `5e-5` and a 2,048-token teacher-turn limit, replacing the previous
`5e-7` / 1,024-token run. Absolute changes are computed before rounding.

The `reward-v4-all-id-fork775` checkpoint continues from 775 completed updates
to 1,000 total updates (not 1,000 additional updates). Its full evaluation is
complete: all nine tasks contain 11,602 predictions in total, and all four
Ped-RM tasks have scores for every sample. Evaluation uses temperature 0, seed
42, and a 2,048-token output limit. The lower Pedagogy Average (0.376 versus
0.525 for the base model) is retained as measured. Per-sample inputs and outputs
are saved in each task's `predictions.jsonl`.

The official MathTutorBench leaderboard uses Solution Correctness F1 and Mistake
Location Micro-F1. The local values in this table are therefore taken from those
specific task metrics rather than the incorrectly selected Accuracy and Macro-F1
fields in the current consolidated local summary.

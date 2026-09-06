# MathTutorBench Comparison

All metrics are reported on a 0–1 scale and higher is better. `Pedagogy Average`
is the arithmetic mean of the four Ped-RM win rates and is not an official
MathTutorBench metric.

| Source | Model | Problem Solving (Accuracy) | Socratic Questioning (BLEU) | Solution Correctness (F1) | Mistake Location (Micro-F1) | Mistake Correction (Accuracy) | Scaffolding Generation (Ped-RM Win Rate) | Pedagogical Instruction Following (Ped-RM Win Rate) | Scaffolding Generation [Hard] (Ped-RM Win Rate) | Pedagogical Instruction Following [Hard] (Ped-RM Win Rate) | Pedagogy Average |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Current MathTutorBench leaderboard | `Qwen/Qwen2.5-7B-Instruct` | 0.85 | 0.19 | 0.64 | 0.49 | 0.52 | 0.44 | 0.40 | 0.42 | 0.44 | 0.425 |
| Current MathTutorBench leaderboard | `eth-nlped/TutorRL-7B` | 0.77 | 0.23 | 0.65 | 0.36 | 0.75 | 0.53 | 0.70 | 0.55 | 0.66 | 0.610 |
| Local evaluation | `Qwen/Qwen3-8B` | 0.921 | 0.253 | 0.685 | 0.371 | 0.675 | 0.331 | 0.740 | 0.324 | 0.706 | 0.525 |
| Local evaluation | `20260905_144444_0904-pedagogical-rl-qwen3-8b-8gpu@globalstep999` | 0.925 | 0.250 | 0.685 | 0.384 | 0.671 | 0.334 | 0.733 | 0.330 | 0.700 | 0.524 |
| Local evaluation | `20260901_182144_0901-preference-v3-reward-v3-none-8gpu@globalstep999` | 0.926 | 0.232 | 0.691 | 0.262 | 0.635 | 0.670 | 0.811 | 0.633 | 0.780 | 0.724 |
| Local evaluation | `20260901_182029_0901-preference-v3-reward-v3-all-id-8gpu@globalstep999` | 0.929 | 0.227 | 0.690 | 0.289 | 0.692 | 0.661 | 0.843 | 0.606 | 0.829 | 0.734 |

## Within-base changes

| Method | Base Pedagogy Average | Trained Pedagogy Average | Absolute Change |
|---|---:|---:|---:|
| `Qwen/Qwen2.5-7B-Instruct` → `eth-nlped/TutorRL-7B` (current MathTutorBench leaderboard) | 0.425 | 0.610 | +0.185 |
| `Qwen/Qwen3-8B` → `PedagogicalRL@globalstep999` (local evaluation) | 0.525 | 0.524 | -0.001 |
| `Qwen/Qwen3-8B` → `reward-v3-none@globalstep999` (local evaluation) | 0.525 | 0.724 | +0.199 |
| `Qwen/Qwen3-8B` → `20260901_182029_0901-preference-v3-reward-v3-all-id-8gpu@globalstep999` (local evaluation) | 0.525 | 0.734 | +0.209 |

## Sources and comparability

- [Current MathTutorBench leaderboard](https://github.com/eth-lre/mathtutorbench)
- [Local `Qwen/Qwen3-8B` results](results/Qwen3-8B-base/summary.yaml)
- [Local `PedagogicalRL@globalstep999` results](results/20260905_144444_0904-pedagogical-rl-qwen3-8b-8gpu/epoch21epochstep12globalstep999/summary.yaml)
- [Local `reward-v3-none@globalstep999` results](results/20260901_182144_0901-preference-v3-reward-v3-none-8gpu/epoch21epochstep12globalstep999/summary.yaml)
- [Local trained-checkpoint results](results/20260901_182029_0901-preference-v3-reward-v3-all-id-8gpu/epoch21epochstep12globalstep999/summary.yaml)

The two Qwen2.5 rows use the current official MathTutorBench leaderboard values.
The four Qwen3 rows were evaluated with the same local runner. PedagogicalRL and
our methods were trained from the same `Qwen/Qwen3-8B` base model for the same
1,000-step training budget, while retaining each method's native training
protocol.

The official MathTutorBench leaderboard uses Solution Correctness F1 and Mistake
Location Micro-F1. The local values in this table are therefore taken from those
specific task metrics rather than the incorrectly selected Accuracy and Macro-F1
fields in the current consolidated local summary.

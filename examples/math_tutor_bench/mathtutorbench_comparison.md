# MathTutorBench Comparison

All metrics are reported on a 0–1 scale and higher is better. `Pedagogy Average`
is the arithmetic mean of the four Ped-RM win rates and is not an official
MathTutorBench metric.

Local Solution Correctness and Mistake Correction use `stepverify-v2` scoring
(2026-09-12): final explicit Yes/No judgment and complete visible correction
response without `Problem:`/`Student:` truncation. All nine local models were
rescored from saved raw outputs; no generation or Ped-RM scoring was rerun.
External leaderboard rows remain official values and are **not protocol-matched
on these two columns**. Original results remain at the source links below;
revised reports are in sibling directories suffixed `-stepverify-v2` and include
`rescore.json` with old/new metrics and source hashes. Other columns are unchanged.

| Source | Model | Problem Solving (Accuracy) | Socratic Questioning (BLEU) | Solution Correctness (F1) | Mistake Location (Micro-F1) | Mistake Correction (Accuracy) | Scaffolding Generation (Ped-RM Win Rate) | Pedagogical Instruction Following (Ped-RM Win Rate) | Scaffolding Generation [Hard] (Ped-RM Win Rate) | Pedagogical Instruction Following [Hard] (Ped-RM Win Rate) | Pedagogy Average |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Current MathTutorBench leaderboard | `Qwen/Qwen2.5-7B-Instruct` | 0.85 | 0.19 | 0.64 | 0.49 | 0.52 | 0.44 | 0.40 | 0.42 | 0.44 | 0.425 |
| Current MathTutorBench leaderboard | `eth-nlped/TutorRL-7B` | 0.77 | 0.23 | 0.65 | 0.36 | 0.75 | 0.53 | 0.70 | 0.55 | 0.66 | 0.610 |
| Local evaluation | `Qwen/Qwen3-8B` | 0.921 | 0.253 | 0.685 | 0.371 | 0.749 | 0.331 | 0.740 | 0.324 | 0.706 | 0.525 |
| Local evaluation | `0906-pedagogical-rl-qwen3-8b-lr5e-5-8gpu@globalstep999` | 0.927 | 0.245 | 0.689 | 0.330 | 0.775 | 0.497 | 0.738 | 0.373 | 0.774 | 0.596 |
| Local evaluation | `20260901_182144_0901-preference-v3-reward-v3-none-8gpu@globalstep999` | 0.926 | 0.232 | 0.693 | 0.262 | 0.787 | 0.670 | 0.811 | 0.633 | 0.780 | 0.724 |
| Local evaluation | `20260901_182029_0901-preference-v3-reward-v3-all-id-8gpu@globalstep999` | 0.929 | 0.227 | 0.736 | 0.289 | 0.770 | 0.661 | 0.843 | 0.606 | 0.829 | 0.734 |
| Local evaluation | `20260901_182029_0901-preference-v3-reward-v3-all-id-8gpu@globalstep1499` | 0.932 | 0.219 | 0.735 | 0.303 | 0.774 | 0.721 | 0.903 | 0.667 | 0.872 | 0.791 |
| Local evaluation | `20260901_182422_0901-preference-v3-reward-v3-subgoal-decomposition-8gpu@globalstep1499` | 0.929 | 0.247 | 0.709 | 0.176 | 0.770 | 0.624 | 0.761 | 0.523 | 0.749 | 0.664 |
| Local evaluation | `20260901_182004_0901-preference-v3-reward-v3-attempt-diagnosis-8gpu@globalstep1499` | 0.920 | 0.277 | 0.734 | 0.282 | 0.741 | 0.490 | 0.719 | 0.385 | 0.700 | 0.574 |
| Local evaluation | `20260901_181950_0901-preference-v3-reward-v3-contrastive-comparison-8gpu@globalstep1499` | 0.925 | 0.244 | 0.694 | 0.347 | 0.705 | 0.653 | 0.872 | 0.544 | 0.838 | 0.727 |
| Local evaluation | `20260908_0901-reward-v4-all-id-fork775-8gpu@globalstep999` | 0.935 | 0.198 | 0.713 | 0.372 | 0.814 | 0.438 | 0.303 | 0.330 | 0.434 | 0.376 |

## Within-base changes

| Method | Base Pedagogy Average | Trained Pedagogy Average | Absolute Change |
|---|---:|---:|---:|
| `Qwen/Qwen2.5-7B-Instruct` → `eth-nlped/TutorRL-7B` (current MathTutorBench leaderboard) | 0.425 | 0.610 | +0.185 |
| `Qwen/Qwen3-8B` → `PedagogicalRL@globalstep999` (local evaluation) | 0.525 | 0.596 | +0.070 |
| `Qwen/Qwen3-8B` → `reward-v3-none@globalstep999` (local evaluation) | 0.525 | 0.724 | +0.199 |
| `Qwen/Qwen3-8B` → `20260901_182029_0901-preference-v3-reward-v3-all-id-8gpu@globalstep999` (local evaluation) | 0.525 | 0.734 | +0.209 |
| `Qwen/Qwen3-8B` → `20260901_182029_0901-preference-v3-reward-v3-all-id-8gpu@globalstep1499` (local evaluation; 1,500 steps) | 0.525 | 0.791 | +0.265 |
| `Qwen/Qwen3-8B` → `20260901_182422_0901-preference-v3-reward-v3-subgoal-decomposition-8gpu@globalstep1499` (local evaluation; 1,500 steps) | 0.525 | 0.664 | +0.139 |
| `Qwen/Qwen3-8B` → `20260901_182004_0901-preference-v3-reward-v3-attempt-diagnosis-8gpu@globalstep1499` (local evaluation; 1,500 steps) | 0.525 | 0.574 | +0.048 |
| `Qwen/Qwen3-8B` → `20260901_181950_0901-preference-v3-reward-v3-contrastive-comparison-8gpu@globalstep1499` (local evaluation; 1,500 steps) | 0.525 | 0.727 | +0.201 |
| `Qwen/Qwen3-8B` → `20260908_0901-reward-v4-all-id-fork775-8gpu@globalstep999` (local evaluation) | 0.525 | 0.376 | -0.149 |

## Sources and comparability

- [Current MathTutorBench leaderboard](https://github.com/eth-lre/mathtutorbench)
- [Local `Qwen/Qwen3-8B` results](results/Qwen3-8B-base/summary.yaml)
- [Local `PedagogicalRL@globalstep999` results](results/0906-pedagogical-rl-qwen3-8b-lr5e-5-8gpu/epoch21epochstep12globalstep999/summary.yaml)
- [Local `reward-v3-none@globalstep999` results](results/20260901_182144_0901-preference-v3-reward-v3-none-8gpu/epoch21epochstep12globalstep999/summary.yaml)
- [Local trained-checkpoint results](results/20260901_182029_0901-preference-v3-reward-v3-all-id-8gpu/epoch21epochstep12globalstep999/summary.yaml)
- [Local `reward-v3-all-id@globalstep1499` results (1,500 steps)](results/20260901_182029_0901-preference-v3-reward-v3-all-id-8gpu/epoch31epochstep42globalstep1499/summary.yaml)
- [Local `reward-v3-subgoal-decomposition@globalstep1499` results (1,500 steps)](results/20260901_182422_0901-preference-v3-reward-v3-subgoal-decomposition-8gpu/epoch31epochstep42globalstep1499/summary.yaml)
- [Local `reward-v3-attempt-diagnosis@globalstep1499` results (1,500 steps)](results/20260901_182004_0901-preference-v3-reward-v3-attempt-diagnosis-8gpu/epoch31epochstep42globalstep1499/summary.yaml)
- [Local `reward-v3-contrastive-comparison@globalstep1499` results (1,500 steps)](results/20260901_181950_0901-preference-v3-reward-v3-contrastive-comparison-8gpu/epoch31epochstep42globalstep1499/summary.yaml)
- [Local `reward-v4-all-id-fork775@globalstep999` results](results/20260908_0901-reward-v4-all-id-fork775-8gpu/epoch21epochstep12globalstep999/summary.yaml)

The two Qwen2.5 rows use the current official MathTutorBench leaderboard values.
The ten Qwen3 rows were evaluated with the same local runner. PedagogicalRL and
our methods were trained from the same `Qwen/Qwen3-8B` base model, while retaining
each method's native training protocol. The `globalstep999` checkpoints share
a 1,000-step training budget. The All, Subgoal, Attempt, and Contrastive `globalstep1499` checkpoints
all have 1,500 completed updates: they are matched-budget with each other,
but not with the 1,000-step checkpoints.

All four 1,500-step evaluations are complete for all nine tasks (10,602 predictions
each) and all four Ped-RM scores, using the same benchmark revision and Ped-RM
model, temperature 0, seed 42, and a 2,048-token output limit. All's Pedagogy
Average is 0.791, compared with 0.734 at 1,000 steps and Subgoal's 0.664 at
1,500 steps. With stepverify-v2 scoring, All leads Subgoal on eight of nine
metrics, including all four Ped-RM metrics; Subgoal leads on Socratic
Questioning BLEU. All's advantage in Pedagogical Instruction Following is 14.3
percentage points on the standard task and 12.2 points on Hard; its Pedagogy
Average advantage is 12.6 points. These win rates compare each model against
the common benchmark baseline, not directly against each other.

Attempt-1500 has a Pedagogy Average of 0.574 (+0.048 over the base model,
computed before rounding). It leads All-1500 and Subgoal-1500 on Socratic
Questioning BLEU. Under stepverify-v2 it trails All on Solution Correctness F1
and both All and Subgoal on Mistake Correction Accuracy; it trails both on all
four Ped-RM metrics.

Contrastive-1500 has a Pedagogy Average of 0.727 (+0.201 over the base model,
computed before rounding), above Subgoal-1500 and Attempt-1500 but below
All-1500. All leads Contrastive on all four Ped-RM metrics, by 3.1 percentage
points on standard Pedagogical Instruction Following and 3.4 points on Hard.
Contrastive has the highest Mistake Location Micro-F1 among these four
1,500-step checkpoints (0.347).

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
specific task metrics. The summary writer now selects these fields as well;
original summaries are preserved, while stepverify-v2 summaries contain the
corrected field selection.

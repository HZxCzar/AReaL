# reward-gt-2

Default training budget: 1500 total steps.

ID: none, attempt-diagnosis, subgoal-decomposition, contrastive-comparison.
The all arm samples these four profiles. OOD: step-demonstration,
causal-justification, independent-verification.

Inherits reward-v3: turn-level leave-one-out baseline, with explain_ratio 1.0,
and the default token loss weighting. Presolve training is disabled; ordinary
verified pre-solving remains enabled with three first-success attempts and a
4096-token budget. Solver responses do not enter training.

Teaching budget: 1024 tokens. The post-std soft-overlong penalty increases
linearly from 0 at 512 tokens to -0.5 at 1024 tokens. The separate format penalty
remains -0.5. This teaching length penalty does not apply to presolve responses.

Launch on allocated GPUs from the repository root:

```bash
bash examples/tutor/run_offline.sh 8 examples/tutor/configs/math/0901/8gpu/reward-gt-2/ID/all.yaml
```

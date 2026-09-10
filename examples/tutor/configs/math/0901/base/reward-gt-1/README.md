# reward-gt-1

Default training budget: 1500 total steps.

ID: none, attempt-diagnosis, subgoal-decomposition, contrastive-comparison.
The all arm samples these four profiles. OOD: step-demonstration,
causal-justification, independent-verification.

Preserves the former reward-gt settings: reward-v4 episode-level leave-one-out
baseline, explain_ratio 1.0, and trained ground-truth pre-solves. Four independent
solver candidates receive correctness rewards; the first correct candidate is
shared with teaching rollouts. Mixed PPO averages the complete loss per response
(actor.loss_weighting: turn). Presolve budget is 4096 tokens.

Teaching budget: 1024 tokens. The post-std soft-overlong penalty increases
linearly from 0 at 512 tokens to -0.5 at 1024 tokens. The separate format penalty
remains -0.5. This teaching length penalty does not apply to presolve responses.

Launch on allocated GPUs from the repository root:

```bash
bash examples/tutor/run_offline.sh 8 examples/tutor/configs/math/0901/8gpu/reward-gt-1/ID/all.yaml
```

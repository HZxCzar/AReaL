# reward-gt

Inherits reward-v4 teaching rewards. Launchable Qwen configs mirror its eight
ID/OOD arms under `0901/8gpu/reward-gt/`.

- `teacher_pre.train: true` enables fixed-count independent solver sampling;
  `teacher_pre.attempts: 4` is the candidate count, not first-success retries.
- Each candidate receives binary correctness and leave-one-out advantage without
  std normalization. All-correct and all-wrong groups have zero solve advantage.
- The first correct candidate is shared with teaching rollouts. Only group slot
  zero exports the four solver responses. All-wrong groups retain solver rows
  but skip teaching. Generation/judge service failures invalidate the group.
- Solver rows never enter teaching returns, baselines or normalization. There is
  no auxiliary lambda. Mixed batches average each response's complete PPO loss
  over its generated tokens, then average responses, including across microbatches.
  Sample counts therefore still set implicit task weights.
- Teaching: 1024 generated tokens, linear post-std length penalty from 0 at 512
  to -0.1 at 1024. Presolve: 4096 tokens, without that teaching penalty.
- Evaluation retains ordinary first-success solver retries and exports no solve
  training rows. Presolve RL currently requires versioned grouped LoRA rollouts,
  REBN, no reward normalization, and no other auxiliary objectives (OPD/world
  model/context/diversity).

The global default remains `teacher_pre.train: false`, teacher budget 2048,
and length penalty 0 at 1536 to -0.05 at 2048. Historical reward-v3/v4/v5 are
unchanged by default. Enabling the train switch opts into response-mean PPO.

From the repository root, on allocated GPUs:

```bash
bash examples/tutor/run_offline.sh 8 examples/tutor/configs/math/0901/8gpu/reward-gt/ID/all.yaml
```

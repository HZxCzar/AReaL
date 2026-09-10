# reward-gt-0

Standalone configuration snapshot from:
`20260901_182029_0901-preference-v3-reward-v3-all-id-8gpu`,
saved at the September 5 continuation from 1000 to 1500 completed steps.

All training settings, four expanded student profiles and eight-GPU allocation
are stored in this directory's default.yaml. There is no inheritance from the
shared default, reward-v3, student pools or alloc.yaml.

ID entries: all, none, attempt-diagnosis, subgoal-decomposition,
contrastive-comparison. Each single-profile entry keeps the same training
settings, selects one frozen student profile and sets its weight to 1.0.
All entries target 1500 steps.

- 1500 total steps, teacher budget 1024 tokens.
- Turn-level leave-one-out baseline; singleton fallback false.
- Explain ratio 1.0; token loss weighting.
- Ordinary verified presolve: three attempts, 4096 tokens; no presolve training.
- No soft-overlong penalty; format/repeat penalties retain historical values.

Operational changes from the snapshot: a new trial name with interpolated
nested trial/output references, and recover.mode auto instead of on so a new run
can start without a recovery checkpoint. Legacy-off options absent from the old
schema (presolve training and soft-overlong shaping) are explicitly disabled.
The historical W&B group is retained.

Launch from the worktree root on allocated GPUs:

```bash
bash examples/tutor/run_offline.sh 8 examples/tutor/configs/math/0901/8gpu/reward-gt-0/ID/all.yaml
```

This freezes configuration, not historical Python code, datasets or the contents
of referenced prompt files. Those resources remain at their configured paths.

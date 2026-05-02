
# configuration
SFT_MODEL_PATH=...
ADDITIONAL_KWARGS="env_kwargs.scenario=simple teacher_obs_kwargs.use_global_obs=true teacher_obs_kwargs.use_individual_thoughts=true"
TRIAL_NAME=...

# launch qwen3 32b as teacher
python -m areal.infra.launcher.slurm examples/hanabi/hanabi_grpo.py     --config examples/hanabi/hanabi_grpo.yaml     experiment_name=gjx-hnb     trial_name=qwne3-32b-teacher actor.path=/storage/openpsi/models/Qwen__Qwen3-32B cluster.n_nodes=4 actor.backend=fsdp:d8p1t4 rollout.backend=sglang:d8p1t4

# record teacher server addresses
TEACHER_SERVER_ADDRS=... 

# launch rl training
python -m areal.infra.launcher.slurm examples/hanabi/hanabi_grpo.py     --config examples/hanabi/hanabi_grpo.yaml     experiment_name=gjx-hnb     trial_name=${TRIAL_NAME} actor.path=${SFT_MODEL_PATH} cluster.n_nodes=4 actor.backend=fsdp:d8p1t2 rollout.backend=sglang:d4p2t2 ${ADDITIONAL_KWARGS}
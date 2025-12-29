# Werewolf Training Script Example

This guide shows how to launch Werewolf RL training with teacher supervision.

## Configuration

```bash
# Model paths
SFT_MODEL_PATH=/storage/openpsi/models/Qwen__Qwen2.5-3B-Instruct
TEACHER_MODEL_PATH=/storage/openpsi/models/Qwen__Qwen3-32B

# Training configuration
TRIAL_NAME=werewolf-rl-trial
EXPERIMENT_NAME=werewolf-grpo

# Werewolf game configuration
ROLE=villager  # or "werewolf" or "both"
NUM_VILLAGERS=2
NUM_WEREWOLVES=3
NUM_WITCHES=1
NUM_FORESEERS=1
NUM_HUNTERS=1
MAX_TURNS=90

# Teacher observation configuration
TEACHER_OBS_KWARGS="teacher_obs_kwargs.use_global_obs=true teacher_obs_kwargs.use_individual_thoughts=true"
```

## Step 1: Launch Teacher Server

First, launch a teacher model server that will provide privileged supervision:

```bash
python -m areal.launcher.slurm examples/werewolf/werewolf_grpo.py \
    --config examples/werewolf/werewolf_grpo.yaml \
    experiment_name=${EXPERIMENT_NAME} \
    trial_name=teacher-server \
    actor.path=${TEACHER_MODEL_PATH} \
    cluster.n_nodes=4 \
    allocation_mode=sglang:d8p1t4
```

**Record the teacher server addresses** from the output for use in the next step:
```bash
TEACHER_SERVER_ADDRS=10.0.0.1:30000,10.0.0.2:30000,10.0.0.3:30000,10.0.0.4:30000
```

## Step 2: Launch RL Training with Teacher

Launch the main RL training with teacher supervision:

```bash
python -m areal.launcher.slurm examples/werewolf/werewolf_grpo.py \
    --config examples/werewolf/werewolf_grpo.yaml \
    experiment_name=${EXPERIMENT_NAME} \
    trial_name=${TRIAL_NAME} \
    actor.path=${SFT_MODEL_PATH} \
    teacher_server_addrs=${TEACHER_SERVER_ADDRS} \
    role=${ROLE} \
    num_villagers=${NUM_VILLAGERS} \
    num_werewolves=${NUM_WEREWOLVES} \
    num_witches=${NUM_WITCHES} \
    num_foreseers=${NUM_FORESEERS} \
    num_hunters=${NUM_HUNTERS} \
    max_turns=${MAX_TURNS} \
    cluster.n_nodes=4 \
    allocation_mode=sglang:d8p1t2+d4p2t2 \
    ${TEACHER_OBS_KWARGS}
```

### Allocation Mode Explanation
- `sglang:d8p1t2+d4p2t2`:
  - First group (`d8p1t2`): 8 GPUs with data parallel size 1, tensor parallel size 2 (student training)
  - Second group (`d4p2t2`): 4 GPUs with data parallel size 2, tensor parallel size 2 (student rollout)

## Alternative: Using API-based Teacher

If you want to use an external API (e.g., GPT-4 or Claude) as the teacher instead of a local server:

```bash
python -m areal.launcher.slurm examples/werewolf/werewolf_grpo.py \
    --config examples/werewolf/werewolf_grpo.yaml \
    experiment_name=${EXPERIMENT_NAME} \
    trial_name=${TRIAL_NAME} \
    actor.path=${SFT_MODEL_PATH} \
    teacher_api_key=${OPENAI_API_KEY} \
    teacher_api_model=gpt-4o-2024-11-20 \
    role=${ROLE} \
    num_villagers=${NUM_VILLAGERS} \
    num_werewolves=${NUM_WEREWOLVES} \
    num_witches=${NUM_WITCHES} \
    num_foreseers=${NUM_FORESEERS} \
    num_hunters=${NUM_HUNTERS} \
    max_turns=${MAX_TURNS} \
    cluster.n_nodes=2 \
    allocation_mode=sglang:d8p1t1+d8p1t1 \
    ${TEACHER_OBS_KWARGS}
```

## Configuration Options

### Teacher Observation Configuration

Control what privileged information the teacher sees:

```bash
# Full privileged information (default)
teacher_obs_kwargs.use_global_obs=true teacher_obs_kwargs.use_individual_thoughts=true

# Only global observation, no agent thoughts
teacher_obs_kwargs.use_global_obs=true teacher_obs_kwargs.use_individual_thoughts=false

# No privileged information (ablation study)
teacher_obs_kwargs.use_global_obs=false teacher_obs_kwargs.use_individual_thoughts=false
```

### Role Configuration

- `role=villager`: Train only when playing as villager team
- `role=werewolf`: Train only when playing as werewolf team
- `role=both`: Train for both roles

### Game Configuration

Adjust the game setup:
```bash
num_villagers=2        # Number of regular villagers
num_werewolves=3       # Number of werewolves
num_witches=1          # Number of witches (0 or 1)
num_foreseers=1        # Number of foreseers (0 or 1)
num_hunters=1          # Number of hunters (0 or 1)
max_turns=90           # Maximum turns per game
turn_discount=1.0      # Reward discount factor per turn
```

## Opponent Configuration

Train against a specific opponent model:

```bash
# Using opponent server
OPP_SERVER_ADDRS=10.0.1.1:30000,10.0.1.2:30000
python -m areal.launcher.slurm examples/werewolf/werewolf_grpo.py \
    ... \
    opp_server_addrs=${OPP_SERVER_ADDRS} \
    opp_tokenizer_path=${OPP_MODEL_PATH}

# Using opponent API
python -m areal.launcher.slurm examples/werewolf/werewolf_grpo.py \
    ... \
    opp_api_key=${OPENAI_API_KEY} \
    opp_api_model=gpt-4o-2024-11-20
```

## Monitoring

Track training progress using WandB:
```bash
stats_logger.wandb.mode=online
```

Logs and generated game traces are saved to:
```
${cluster.fileroot}/logs/${experiment_name}/${trial_name}/generated/
```

## Example Complete Command

```bash
python -m areal.launcher.slurm examples/werewolf/werewolf_grpo.py \
    --config examples/werewolf/werewolf_grpo.yaml \
    experiment_name=my-werewolf-exp \
    trial_name=villager-3b-with-teacher \
    actor.path=/storage/openpsi/models/Qwen__Qwen2.5-3B-Instruct \
    teacher_server_addrs=10.0.0.1:30000,10.0.0.2:30000 \
    role=villager \
    num_villagers=2 \
    num_werewolves=3 \
    num_witches=1 \
    num_foreseers=1 \
    num_hunters=1 \
    max_turns=90 \
    turn_discount=1.0 \
    cluster.n_nodes=4 \
    allocation_mode=sglang:d8p1t2+d4p2t2 \
    teacher_obs_kwargs.use_global_obs=true \
    teacher_obs_kwargs.use_individual_thoughts=true \
    stats_logger.wandb.mode=online
```

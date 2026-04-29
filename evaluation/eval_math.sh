# export UV_DEFAULT_INDEX="https://artifacts.antgroup-inc.cn/simple/"
# cd /storage/openpsi/users/xmy/inclusionAI/AReaL-New


nohup python eval_and_aggregate.py \
    --model_path /storage/openpsi/experiments/checkpoints/admin/xmy-werewolf-eval/qwen3-8b-prm2.5-vs-claude-4-5-thinking-noqa-base/default/epoch0epochstep0globalstep0 \
    --output_path eval_res/qwen3-8b \
    --max_gen_tokens 32768 \
    --data_names math_500,aime24,amc23 \
    --prompt_type qwen3-think \
    --task math &> eval_and_aggregate_parallel.log &

python eval_naive.py \
    --model /storage/openpsi/experiments/checkpoints/admin/xmy-werewolf-eval/qwen3-8b-prm2.5-vs-claude-4-5-thinking-noqa-base/default/epoch0epochstep0globalstep0 \
    --data data/amc23/test.jsonl \
    --template qwen3-think \
    --tensor-parallel-size 8 \
    --batch-size 16 \
    --max-tokens 32768 \
    --output-dir outputs/qwen3_amc23

python eval_naive.py \
    --model /storage/openpsi/experiments/checkpoints/admin/xmy-werewolf-eval/qwen3-8b-prm2.5-vs-claude-4-5-thinking-noqa-step149/default/epoch0epochstep0globalstep0 \
    --data data/amc23/test.jsonl \
    --template qwen3-think \
    --tensor-parallel-size 8 \
    --batch-size 16 \
    --max-tokens 32768 \
    --output-dir outputs/qwen3_amc23_step149

python eval_naive.py \
    --model /storage/openpsi/experiments/checkpoints/admin/xmy-werewolf-eval/qwen3-8b-prm2.5-vs-claude-4-5-thinking-noqa-step4/default/epoch0epochstep0globalstep0 \
    --data data/aime24/test.jsonl \
    --template qwen3-think \
    --tensor-parallel-size 8 \
    --batch-size 16 \
    --max-tokens 32768 \
    --output-dir outputs/qwen3_aime24_step4

# qwen25-math-cot
# /storage/openpsi/experiments/checkpoints/admin/gjx-hnb/simple-7b-sft-prm0.5-rl-act-no-qa-train-trial1/default/epoch0epochstep21globalstep21
# epoch2epochstep25globalstep149
# epoch4epochstep51globalstep299

python eval_naive.py \
    --model /storage/openpsi/experiments/checkpoints/admin/gjx-hnb/simple-7b-sft-prm0.5-rl-act-no-qa-train-trial1/default/epoch4epochstep51globalstep299 \
    --data data/aime25/test.jsonl \
    --template qwen25-math-cot \
    --tensor-parallel-size 4 \
    --batch-size 16 \
    --max-tokens 32768 \
    --output-dir outputs/qwen25_aime25_step299
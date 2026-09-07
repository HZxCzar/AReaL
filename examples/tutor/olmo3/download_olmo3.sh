#!/usr/bin/env bash
# Stage allenai/Olmo-3-7B-Instruct into the shared offline HF cache.
#
# Why this exists rather than a bare `hf download`:
#   - The training launcher (examples/tutor/run_official.sh) exports
#     HF_HUB_OFFLINE=1. This is the one step that needs the network, so the
#     script clears the offline vars explicitly instead of trusting the shell
#     it happens to be launched from.
#   - It pins HF_HUB_CACHE to the single shared cache that already holds
#     Qwen3-8B, so the model lands where the configs' actor.path lives.
#   - It verifies afterwards that the three config fields sglang's olmo2.py
#     reads (layer_types / sliding_window / rope_scaling) are actually there.
#     The whole plan to serve OLMo 3 on the olmo2 implementation rests on them,
#     so a silent absence here would surface much later as wrong numbers.
#
# Read-only with respect to the running training jobs. Safe to re-run: the HF
# cache is content-addressed and resumes partial downloads.
set -euo pipefail

REPO="allenai/Olmo-3-7B-Instruct"
# Resolves to /inspire/hdd/project/qproject-fundationmodel/public/wxxu/.cache/huggingface/hub
CACHE="/inspire/qb-ilm/project/qproject-fundationmodel/public/wxxu/hdd/.cache/huggingface/hub"
VENV="/inspire/qb-ilm/project/qproject-fundationmodel/public/wxxu/TAgent/AReaL/.venv"

export HF_HUB_CACHE="$CACHE"
# MUST be off for this step.
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE HF_DATASETS_OFFLINE
export HF_HUB_ENABLE_HF_TRANSFER=0

if [[ ! -d "$CACHE" ]]; then
  echo "Cache dir missing: $CACHE" >&2
  exit 1
fi

# 3 bf16 shards ~= 15GB. The gpfs_hdd volume reads 100% used with ~3.7T free,
# and the tutor runs are writing to it, so refuse to start on a thin margin.
AVAIL_GB=$(df -BG --output=avail "$CACHE" | tail -1 | tr -dc '0-9')
if [[ -z "$AVAIL_GB" ]] || (( AVAIL_GB < 40 )); then
  echo "Only ${AVAIL_GB:-?}GB free on $CACHE; need ~15GB plus headroom." >&2
  exit 1
fi

echo "repo    : $REPO"
echo "cache   : $CACHE"
echo "free    : ${AVAIL_GB}GB"

# shellcheck disable=SC1091
source "$VENV/bin/activate"

echo
echo "== downloading =="
# hf download resumes; retry for transient network faults.
for attempt in 1 2 3; do
  if hf download "$REPO" --cache-dir "$CACHE"; then
    break
  fi
  if (( attempt == 3 )); then
    echo "download failed after 3 attempts" >&2
    exit 1
  fi
  echo "attempt $attempt failed; retrying in 15s" >&2
  sleep 15
done

echo
echo "== verifying =="
python3 - "$CACHE" <<'PYEOF'
import glob, json, os, sys

cache = sys.argv[1]
pattern = os.path.join(
    cache, "models--allenai--Olmo-3-7B-Instruct", "snapshots", "*"
) + os.sep
snaps = [s for s in glob.glob(pattern) if os.path.exists(os.path.join(s, "config.json"))]
if not snaps:
    print("FAIL: no snapshot containing config.json")
    sys.exit(1)

snap = snaps[0]
cfg = json.load(open(os.path.join(snap, "config.json")))
ok = True

print("snapshot      :", snap.rstrip(os.sep))
print("model_type    :", cfg.get("model_type"))
print("architectures :", (cfg.get("architectures") or ["?"])[0])
print("ctx           :", cfg.get("max_position_embeddings"))
print("hidden/layers :", cfg.get("hidden_size"), "/", cfg.get("num_hidden_layers"))

if cfg.get("model_type") != "olmo3":
    print("FAIL: expected model_type 'olmo3'")
    ok = False

# The three fields sglang/srt/models/olmo2.py reads to drive the sliding-window
# split and to apply YaRN on full-attention layers only.
layer_types = cfg.get("layer_types")
if not layer_types:
    print("FAIL: no layer_types -- sglang olmo2.py cannot infer sliding layers")
    ok = False
else:
    n_slide = sum(1 for x in layer_types if x == "sliding_attention")
    print(
        "layer_types   : %d layers, %d sliding / %d full"
        % (len(layer_types), n_slide, len(layer_types) - n_slide)
    )

sliding_window = cfg.get("sliding_window")
print("sliding_window:", sliding_window)
if sliding_window is None:
    print("FAIL: no sliding_window")
    ok = False

rope_scaling = cfg.get("rope_scaling") or {}
print("rope_scaling  :", json.dumps(rope_scaling))
if rope_scaling.get("rope_type") != "yarn":
    print("WARN: rope_type is not 'yarn' -- re-check the sglang rope path")
else:
    # sglang filters rope_scaling for the key 'attn_factor', while OLMo 3 writes
    # 'attention_factor'. The config value is therefore ignored and sglang
    # recomputes mscale = 0.1*ln(factor)+1. Check they still agree; if a future
    # checkpoint hand-tunes attention_factor this is where it diverges silently.
    import math

    factor = rope_scaling.get("factor")
    stated = rope_scaling.get("attention_factor")
    if factor and stated:
        implied = 0.1 * math.log(float(factor)) + 1.0
        print(
            "yarn mscale   : config %.16f vs sglang-recomputed %.16f%s"
            % (stated, implied, "" if abs(stated - implied) < 1e-9 else "   <-- MISMATCH")
        )
        if abs(stated - implied) >= 1e-9:
            print("FAIL: sglang would silently use a different YaRN scale")
            ok = False

# Shard integrity against the index, resolving symlinks into blobs/.
idx_path = os.path.join(snap, "model.safetensors.index.json")
if os.path.exists(idx_path):
    weight_map = json.load(open(idx_path))["weight_map"]
    shards = sorted(set(weight_map.values()))
    total = 0
    for shard in shards:
        p = os.path.join(snap, shard)
        if not os.path.exists(p):
            print("FAIL: missing shard", shard)
            ok = False
        else:
            total += os.path.getsize(os.path.realpath(p))
    print("shards        : %d present, %.1f GB" % (len(shards), total / 1e9))
else:
    print("WARN: no model.safetensors.index.json")

for f in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
    if not os.path.exists(os.path.join(snap, f)):
        print("WARN: missing", f)

print()
print("RESULT:", "OK" if ok else "PROBLEMS FOUND")
print()
print("actor.path / tokenizer_path for the config:")
print(snap.rstrip(os.sep))
sys.exit(0 if ok else 1)
PYEOF

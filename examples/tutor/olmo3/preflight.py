"""Pre-flight for run_olmo3.sh.

One job, and it fails loudly here rather than tens of minutes into a run:

  Confirm the OLMo 3 rope dtype fix is live. If it is not, ppo_update dies on
     a bf16-vs-fp32 gradient-checkpoint metadata mismatch that looks nothing
     like its cause. See sitecustomize.py.

The stop token ids that AReaL would otherwise miss are declared in
configs/math/0901/8gpu/olmo3/base.yaml instead, where they are visible and
reviewable rather than computed at launch.
"""

import json
import os
import sys

log = lambda *a: print(*a, file=sys.stderr)

snapshot = sys.argv[1]

import torch
from transformers import AutoConfig
from transformers.models.olmo3 import modeling_olmo3 as m
from transformers import AutoTokenizer

# --- 1. rope fix -----------------------------------------------------------
if not getattr(m.Olmo3RotaryEmbedding, "_tagent_rope_dtype_fix_applied", False):
    sys.exit("rope fix did NOT apply; sitecustomize.py was not imported")

# Assert the behaviour, not just the marker: build the real full-attention
# rotary (the one that reads the yarn rope_scaling and produced the mismatched
# tensor) and confirm bf16 in gives bf16 out.
cfg = AutoConfig.from_pretrained(snapshot)
rot = m.Olmo3RotaryEmbedding(config=cfg)
x = torch.zeros(1, 8, cfg.hidden_size, dtype=torch.bfloat16)
cos, sin = rot(x, torch.arange(8).unsqueeze(0))
if cos.dtype is not torch.bfloat16 or sin.dtype is not torch.bfloat16:
    sys.exit("rope still returns %s/%s, expected bfloat16" % (cos.dtype, sin.dtype))
log("  rope fix    : bf16 in -> bf16 out, ok")

# Stop token ids are declared by the config now
# (configs/math/0901/8gpu/olmo3/base.yaml), not injected from here.

"""Register Olmo3ForCausalLM onto sglang's existing olmo2 implementation.

sglang 0.5.9 ships no olmo3 module, and olmo2.py ends with
``EntryClass = Olmo2ForCausalLM``, so an OLMo 3 checkpoint -- which declares
``Olmo3ForCausalLM`` -- resolves to nothing native and falls back to the generic
transformers backend. That backend has no LoRA support at all
(sglang/srt/models/transformers.py contains no lora reference), and this
training loop is rank-16 LoRA with hot /load_lora_adapter, so the fallback is
not usable here.

WHY REUSING olmo2 IS CORRECT, not a shortcut. OLMo 3 keeps OLMo 2's block
structure and expresses its own additions through the config, and sglang's
olmo2.py already reads exactly those fields:

  - olmo2.py places sliding-window attention per layer from
    ``config.layer_types[layer_id] == "sliding_attention"``. The downloaded
    Olmo-3-7B-Instruct config carries 32 layer_types, 24 sliding / 8 full in a
    3:1 pattern.
  - It sets ``rope_scaling = config.rope_scaling if sliding_window is None else
    {"rope_type": "default"}`` -- its own comment reads "Rope scaling is only
    applied on full attention layers." That is the OLMo 3 design: YaRN on full
    attention, plain RoPE on the sliding layers. transformers 4.57.1's
    modeling_olmo3.py builds the same split via a two-entry rotary_embs
    ModuleDict, so training and serving agree.
  - Its load_weights stacked_params_mapping fuses q/k/v -> qkv_proj and
    gate/up -> gate_up_proj, which is the form the checkpoint ships. The
    checkpoint's remaining per-layer names (o_proj, down_proj, q_norm, k_norm,
    post_attention_layernorm, post_feedforward_layernorm) are the olmo2 set;
    post_feedforward_layernorm in particular is OLMo 2's signature.

Upstream reached the same conclusion -- sgl-project/sglang#31175, "Add native
OLMo3 support by reusing the Olmo2 implementation" -- it simply has not landed.

WHY THIS LIVES OUTSIDE THE VENV. Every one of the sixteen AReaL worktrees
symlinks .venv to TAgent/AReaL/.venv, so editing olmo2.py in place would put
every sglang launch, including the running reward-v4/v5 arms on their next
restart, behind a one-character typo in a file they all share.
SGLANG_EXTERNAL_MODEL_PACKAGE is read once at registry import
(sglang/srt/models/registry.py:131) and defaults to empty, so anything that
does not set it is untouched.

ONE ENTRY CLASS ONLY -- do not add a second. External packages are registered
with overwrite=True, so an EntryClass here that happened to name a built-in
architecture would silently replace sglang's own implementation of it. Because
this module names only Olmo3ForCausalLM, setting the env var on a Qwen run is
harmless: the registry resolves by the architecture the checkpoint declares, and
a Qwen3 checkpoint still finds sglang's built-in Qwen3ForCausalLM.
"""

from sglang.srt.models.olmo2 import Olmo2ForCausalLM


class Olmo3ForCausalLM(Olmo2ForCausalLM):
    """OLMo 3 served by the OLMo 2 implementation.

    Deliberately empty. Every OLMo 3 difference that sglang needs to honour
    arrives through the config fields olmo2.py already reads; adding anything
    here would be duplicating logic that is one import away.
    """


EntryClass = Olmo3ForCausalLM

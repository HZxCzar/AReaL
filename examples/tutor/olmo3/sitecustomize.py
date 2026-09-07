"""Cast OLMo 3's rotary cos/sin back to the input dtype.

THE OMISSION. transformers 4.57.1's Olmo3RotaryEmbedding.forward ends with

    with torch.autocast(device_type=device_type, enabled=False):   # Force float32
        ...
        return cos, sin

returning float32 and never casting to the input dtype. Every other family in
the same transformers version ends with
``return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)`` -- qwen3 line 332,
llama line 106, gemma3 line 179 -- and does it *outside* the autocast block.
olmo2 line 293 and olmo3 line 316 both return from inside it, so this is an
omission in the OLMo modular definition rather than anything about the OLMo 3
weights.

WHY IT BREAKS TRAINING, exactly. Two things combine:

  * OLMo 3 computes the rope OUTSIDE the checkpointed region. Olmo3Model.forward
    builds position_embeddings from self.rotary_embs (modeling_olmo3.py:410-412)
    and hands them to each decoder layer (line 422), while HF's
    gradient_checkpointing_enable wraps the decoder layer. So cos/sin are an
    *input* to the checkpointed function, not a value produced inside it.
  * AReaL's FSDP runs MixedPrecisionPolicy(param_dtype=bfloat16,
    cast_forward_inputs=True) -- fsdp_utils/parallel.py:384-387 -- so FSDP2
    casts every module forward's inputs to bf16 at the module boundary.

On the forward pass the fp32 cos/sin are therefore cast to bf16 by FSDP, and
that bf16 tensor is what gets saved. On the recompute, torch re-invokes the
inner function directly and bypasses the FSDP boundary hook, so cos/sin stay
fp32. With use_reentrant=False (fsdp_engine.py:1058-1060) the non-reentrant
path compares metadata and raises:

    Recomputed values for the following tensors have different metadata
    saved metadata:      {'shape': [1, 1, 20480, 128], 'dtype': torch.bfloat16}
    recomputed metadata: {'shape': [1, 1, 20480, 128], 'dtype': torch.float32}

qwen3/llama/gemma3 never hit this: their rotary already returns bf16, so FSDP's
cast is a no-op and the two passes agree by construction.

THIS DOES NOT CHANGE THE FORWARD PASS, and that is the point worth being clear
about. Under cast_forward_inputs=True the forward was already receiving bf16
cos/sin; this patch just performs that cast at the source, where the other
families do it, so the recompute sees the same thing.

    before  forward: fp32 -> FSDP casts to bf16 -> q*cos in bf16
    after   forward: bf16 -> FSDP cast is a no-op -> q*cos in bf16   (identical)
    before  recompute: fp32 -> no cast -> q*cos in fp32              (the mismatch)
    after   recompute: bf16 -> q*cos in bf16                         (now agrees)

So this is not a precision reduction, and it changes nothing about the
architecture: layer count, dims, head count, the 24-sliding / 8-full attention
pattern, the yarn factor of 8.0 and its attention_scaling of 1.2079441541679836,
and which layers receive yarn are all untouched, as is the fp32 computation
inside the rotary itself. Only the dtype it hands out changes.

It also means the question of whether OLMo's fp32 rope is deliberate for YaRN
precision is moot here: with this FSDP policy the forward never received fp32
cos to begin with. That question would matter in a training setup that does not
set cast_forward_inputs.

SCOPE. The patch is confined twice over:

  * By process. It does nothing unless TAGENT_OLMO3_ROPE_FIX=1 is set, and only
    run_olmo3.sh sets it. Every existing run, and anything else that happens to
    inherit this PYTHONPATH, is unaffected.
  * By class. Olmo3RotaryEmbedding lives in
    transformers.models.olmo3.modeling_olmo3 and is neither Olmo2RotaryEmbedding
    nor a subclass of it -- verified, including that the two forward attributes
    are distinct function objects. qwen3, llama, gemma3 and olmo2 cannot be
    reached from here.

Nothing is written to the venv; transformers on disk is unmodified.
"""

import os
import sys

_FLAG = "TAGENT_OLMO3_ROPE_FIX"
_APPLIED_ATTR = "_tagent_rope_dtype_fix_applied"


def _apply() -> None:
    import functools

    from transformers.models.olmo3 import modeling_olmo3

    cls = modeling_olmo3.Olmo3RotaryEmbedding
    if getattr(cls, _APPLIED_ATTR, False):
        return

    original = cls.forward

    @functools.wraps(original)
    def forward(self, x, position_ids):
        # `original` still carries its own @torch.no_grad() and
        # @dynamic_rope_update decorators; this only fixes the return dtype.
        cos, sin = original(self, x, position_ids)
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

    cls.forward = forward
    setattr(cls, _APPLIED_ATTR, True)


if os.environ.get(_FLAG) == "1":
    try:
        _apply()
        # stderr, not stdout: this module is imported by every process that
        # inherits the PYTHONPATH, and run_olmo3.sh captures a pre-flight
        # script's stdout as data.
        print("[olmo3-rope-fix] applied to Olmo3RotaryEmbedding.forward",
              file=sys.stderr, flush=True)
    except Exception as exc:  # pragma: no cover
        # Do not raise from sitecustomize: it would abort the interpreter with a
        # traceback that hides the real launch. run_olmo3.sh pre-flights this
        # patch and refuses to start if it did not take, so a loud line here is
        # enough to explain why.
        print(
            "[olmo3-rope-fix] FAILED to apply (%s: %s). Training would die in "
            "ppo_update on a checkpoint metadata mismatch." % (type(exc).__name__, exc),
            file=sys.stderr,
            flush=True,
        )

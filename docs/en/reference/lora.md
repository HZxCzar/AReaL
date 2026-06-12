# LoRA Reference

LoRA is a parameter-efficient fine-tuning technique that injects trainable low-rank
matrices into pre-trained weights, typically around linear layers. Compared with
full-parameter fine-tuning, this reduces memory usage and compute cost substantially,
making RL fine-tuning of large models much more practical on limited hardware.

In AReaL, this is especially useful for:

- reinforcement learning with very large models, including 70B+ models, on relatively
  modest hardware such as 8 x 80 GB GPUs,
- enabling larger batch sizes because LoRA reduces training memory pressure,
- simplifying transfer and deployment because only the LoRA adapters need to be saved
  and shipped,
- \[Future\] fine-tune multiple LoRA adapters more efficiently in parallel for better
  hardware utilization (see RFC
  [#609](https://github.com/inclusionAI/AReaL/issues/609)).

This guide explains how to enable LoRA in RL training and configure the related
parameters.

## Backend Support

The current LoRA support matrix in AReaL is:

| Engine   | vLLM | SGLang |
| -------- | ---- | ------ |
| FSDP2    | ✅   | ✅     |
| Megatron | ✅   | ❌     |
| Archon   | ❌   | ❌     |

Example scripts:

| Engine   | Example script                                |
| -------- | --------------------------------------------- |
| FSDP2    | `examples/math/gsm8k_grpo_lora.yaml`          |
| Megatron | `examples/math/gsm8k_grpo_megatron_lora.yaml` |

## Runtime Version Semantics

AReaL's asynchronous LoRA version management is primarily designed for the
SGLang + disk-update path. In that mode, multiple versioned LoRA adapters may
remain loaded on the rollout servers at the same time, and the runtime tracks
active versions so an adapter is not unloaded while an in-flight workflow can
still request it.

vLLM uses different runtime LoRA options (`max_loras` / `lora_modules`) and its
XCCL LoRA update path behaves like a fixed adapter slot whose weights are
replaced in place. Do not rely on vLLM XCCL LoRA to keep multiple historical
LoRA versions concurrently addressable. Use SGLang + disk updates when stale or
lagged rollout workflows need to request older LoRA versions.

## Core LoRA Parameters

| Parameter         | What it controls                                                                                        | Typical values        |
| ----------------- | ------------------------------------------------------------------------------------------------------- | --------------------- |
| `use_lora`        | Enables LoRA fine-tuning mode.                                                                          | `true` / `false`      |
| `lora_rank` (`r`) | Rank of the low-rank adapters. Higher rank increases capacity and memory/compute cost.                  | `8`, `16`, `32`, `64` |
| `lora_alpha`      | LoRA scaling factor. Effective adapter scale is commonly thought of as proportional to `alpha / r`.     | `16`, `32`, `64`      |
| `target_modules`  | Which model submodules receive LoRA adapters. This is the most important architecture-specific setting. | e.g. \[`all-linear`\] |
| `peft_type`       | PEFT method type. In AReaL configs, this is LoRA.                                                       | `lora`                |

## Practical Notes

- Start with `r=16` or `r=32` for most models, then tune upward only if needed.
- Keep `target_modules` consistent with your model architecture naming.
- Currently only dense models (non MoE) are supported.
- For Megatron backend, LoRA requires `megatron-bridge` instead of `mbridge`.

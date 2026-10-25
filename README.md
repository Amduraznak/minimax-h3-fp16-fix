# MiniMax H3 at native fp16 on pre-bf16 GPUs (V100 / Volta)

MiniMax H3 declares `supported_inference_dtypes = [bf16, fp32]`, so on GPUs
without bf16 support ComfyUI silently falls back to fp32 manual-cast. Forcing
`--fp16-unet` produces pure black frames. This patch makes native fp16 work
with exact math: no clamping, no approximation, no quality change.

Measured on a V100 32GB at 1120x768, 124 frames: **~30 s/step vs 330-370
s/step fp32 (~11x)**. Output is indistinguishable from the fp32 path.

## Why fp16 breaks

Three separate places in H3 exceed fp16's maximum representable value
(65504):

1. **`condition_proj`**: Qwen3-VL encoder hidden states project to ~96k on
   the very first linear layer.
2. **The residual stream**: attention-sink rows grow from ~81k to ~4.3M
   across the 50 DiT blocks. Since the magnitude grows ~50x through the
   stack, no fixed rescale can work.
3. **`attn.out_proj` and MLP `fc2`**: outputs exceed fp16 max at some
   sigmas (fc2 exceeds 2M).

## How the fix works

1. **condition_proj** runs with fp32 input.
2. **The residual stream stays fp32** across all 50 blocks. The
   normalized/modulated branch inputs are cast to fp16 so the attention and
   MLP matmuls still run on fp16 tensor cores, and the gated adds accumulate
   in fp32. The matmuls are ~99% of the compute, which is where the speed
   comes from.
3. **out_proj / fc2 are rescaled**: `out = f(x / K) * K` with K a power of
   two (64 for out_proj, 256 for fc2). The projections are linear and K is a
   power of two, so this is bit-exact; it only shifts the exponent range.

ComfyUI's existing fp32 islands for H3 (`video_out` / `audio_out`, the patch
projections, the AdaLN table) are untouched.

## Which GPUs benefit

- **Volta (V100, Titan V) and Turing (T4, RTX 20xx, Quadro RTX, Titan RTX):
  the big win.** These tensor cores have no bf16/TF32 support, so the fp32
  fallback leaves them idle. fp16 is the only format that uses them. ~11x
  measured on the V100; the exact ratio varies per card. Turing is untested
  but in the same situation, reports welcome.
- **P100:** ~2x expected. No tensor cores, but double-rate vector fp16.
- **P40 / GTX 10-series (sm_61): don't bother.** Those chips run fp16 at
  1/64 rate, so this patch fixes the black frames but will be far slower
  than the fp32 fallback.
- **Ampere and newer:** no benefit. bf16 already runs on the tensor cores
  and the patch stays inert.
- **AMD:** reported working on a V620 under ROCm (fixed the dtype mismatch,
  with a real speedup).

## Install

1. Copy `minimax_h3_fp16_fix.py` into `ComfyUI/custom_nodes/`.
2. Launch ComfyUI with `--fp16-unet`.

The patch self-activates only when the H3 model is instantiated as fp16. In
bf16/fp32 mode it does nothing, so it is safe to leave installed.

## Caveats

- This monkey-patches internal ComfyUI classes (`MiniMaxH3Model`, `DiTBlock`,
  `MLP`). A ComfyUI refactor of `comfy/ldm/minimax/model.py` can break it.
  Tested against ComfyUI v0.30.0.
- Tested on one machine (V100 32GB). The other cards in the benefit table
  are expected to behave the same but are unverified (except the AMD V620,
  which was confirmed by a user).

## Current ComfyUI attention callback and V100 validation

Newer ComfyUI passes an optional `attention` callable to `DiTBlock.forward`.
The patch accepts this argument, honors it in the active FP16 path, and
forwards it to the original method when the FP16 fix is inactive. This avoids
an unexpected-keyword error and preserves attention overrides. The numerical
rescaling logic is unchanged.

This callback adaptation was tested against ComfyUI commit
`1568e6cfd04586a4b3c4e1817ea7dde09b1bf9e7`. Compatibility with older
ComfyUI signatures was not tested by this contribution.

Additional hardware validation: Tesla V100 PCIe 32 GiB (sm_70), 150 W power
limit, Python 3.13.5, PyTorch 2.11.0+cu126, comfy-kitchen 0.2.35. Used the
pruned INT8 ConvRot FL2VA model, Q4_K_M Qwen3-VL text encoder on CPU in FP32,
FP16 video VAE and FP32 audio VAE. Isolated runs used low-VRAM mode,
dynamic VRAM disabled, 10 GiB reserved and PyTorch attention.

Both runs used 512x896, 124 native frames at 24 fps, 20
`res_multistep`/`simple` steps, CFG 1, seed 12345:

| Mode | Full native generation, including conditioning | Peak sampled total GPU memory |
| --- | ---: | ---: |
| Text-to-video | 16 min 49 s | 22.01 GiB |
| Image-to-video | 19 min 25 s | 22.40 GiB |

All 124 native frames and stereo audio decoded successfully. Finite residual
checks passed across all 50 blocks. These observations do not establish
full-model FP32/FP16 equivalence or an independent speedup ratio; runtime and
memory also depend on the surrounding loading/offloading configuration.

A small CUDA regression check uses synthetic components, no model downloads,
and exits nonzero on failure. Run it in a fresh process with the ComfyUI
Python environment (the patch changes classes globally):

```bash
PYTHONPATH=/path/to/ComfyUI /path/to/ComfyUI/venv/bin/python tests/test_attention_override.py
```

It checks unchanged FP32 outputs and callback forwarding, FP16 callback
execution, default/override agreement, FP32 residuals, and finite outputs
with residual inputs beyond the FP16 range. A separate checkout of this
repository is recommended for the check so the patch is imported once.

## AI disclosure

Generative AI (Anthropic's Claude) was used to identify the root cause and
create this fix, driving per-block/per-op numerical probes against a live
V100. All results were verified with real renders on that hardware.

## License

MIT

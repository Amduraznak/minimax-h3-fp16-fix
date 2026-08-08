# MiniMax H3: exact-math fp16 fixes for GPUs without bf16 support (Volta/V100 etc.)
#
# H3's supported_inference_dtypes is [bf16, fp32], so pre-Ampere cards fall back to
# fp32 manual-cast (~11x slower than native fp16 on a V100). Forcing --fp16-unet
# alone produces black frames because three places in the model genuinely exceed
# fp16's max representable value (65504):
#
#   1) condition_proj: Qwen3-VL hidden states project to ~96k.
#      Fix: run the projection with fp32 input.
#   2) The residual stream: attention-sink rows grow from ~81k to ~4.3M across the
#      50 DiT blocks. No fixed rescale can cover that range.
#      Fix: keep the residual stream in fp32; branch matmuls (attn/MLP) stay fp16.
#   3) attn out_proj and MLP fc2 outputs exceed fp16 max at some sigmas.
#      Fix: exact linear rescale out = f(x/K) * K. K is a power of two, and the
#      projections are linear, so this is bit-exact — it just shifts the exponent
#      range. K=64 for out_proj; fc2 needs K=256 (values exceed 2M).
#
# Everything here is exact math — no clamping, no approximation. Output matches
# the fp32 path up to normal fp16 rounding in the matmuls themselves.
#
# Install: drop this file into ComfyUI/custom_nodes/ and launch with --fp16-unet.
# The patches self-activate only when the H3 model is instantiated with
# dtype=torch.float16; in bf16/fp32 mode this file does nothing.

import torch
import comfy.ldm.minimax.model as mm

K_OUT_PROJ = 64.0
K_FC2 = 256.0  # 32 is insufficient — fc2 outputs exceed 2M at some sigmas

_orig_model_init = mm.MiniMaxH3Model.__init__
_orig_mlp_forward = mm.MLP.forward
_orig_block_forward = mm.DiTBlock.forward


def _patched_init(self, *args, **kwargs):
    _orig_model_init(self, *args, **kwargs)
    if self.dtype != torch.float16:
        return

    # (1) condition_proj: cast input to fp32 so the projection can't overflow.
    cp = self.condition_proj
    orig_cp = cp.forward
    cp.forward = lambda t: orig_cp(t.to(torch.float32))

    # (3a) out_proj: exact power-of-two rescale around the projection.
    for block in self.blocks:
        block._h3_fp16_fix = True
        op = block.attn.out_proj
        orig_op = op.forward

        def _safe_out_proj(t, _f=orig_op):
            return _f((t / K_OUT_PROJ).to(torch.float16)).to(torch.float32).mul_(K_OUT_PROJ)

        op.forward = _safe_out_proj


mm.MiniMaxH3Model.__init__ = _patched_init


# (3b) MLP: SwiGLU pointwise math in fp32, down-proj rescaled by K_FC2.
# Gated on input dtype so bf16/fp32 runs take the original path.
def _patched_mlp_forward(self, x):
    if x.dtype != torch.float16:
        return _orig_mlp_forward(self, x)
    u = self.fc1(x)
    gate, up = u.chunk(2, dim=-1)
    s = torch.nn.functional.silu(gate.to(torch.float32)).mul_(up.to(torch.float32))
    return self.fc2((s / K_FC2).to(torch.float16)).to(torch.float32).mul_(K_FC2)


mm.MLP.forward = _patched_mlp_forward


# (2) fp32 residual stream: x stays fp32 across all 50 blocks; the norm+modulate
# results are cast to fp16 for the attn/MLP branches, and gates accumulate in fp32.
def _patched_block_forward(self, x, t_emb, mod_segments, rope_freqs, transformer_options={}):
    if not getattr(self, "_h3_fp16_fix", False):
        return _orig_block_forward(self, x, t_emb, mod_segments, rope_freqs, transformer_options)
    if x.dtype != torch.float32:
        x = x.to(torch.float32)
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaln_proj(t_emb)
    h = mm._mod_scale_shift(self.norm1(x), shift_msa, scale_msa, mod_segments).to(torch.float16)
    att = self.attn(h, rope_freqs=rope_freqs, transformer_options=transformer_options)
    x = mm._mod_gate(x, gate_msa, att.to(torch.float32), mod_segments)
    h = mm._mod_scale_shift(self.norm2(x), shift_mlp, scale_mlp, mod_segments).to(torch.float16)
    m = self.mlp(h)
    return mm._mod_gate(x, gate_mlp, m, mod_segments)


mm.DiTBlock.forward = _patched_block_forward

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

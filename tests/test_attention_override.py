"""Isolated tiny synthetic V100 checks. No real H3 weights or service changes."""
import importlib.util,json,sys,traceback
from pathlib import Path
import torch
import comfy.ops as ops
import comfy.ldm.minimax.model as mm
ROOT=Path(__file__).resolve().parents[1]
results={"scope":"synthetic components only; not full H3 output/VRAM validation","torch":torch.__version__,"gpu":torch.cuda.get_device_name(),"capability":torch.cuda.get_device_capability(),"tests":[]}
torch.manual_seed(123)
def make(dtype):
 model=mm.MiniMaxH3Model(hidden_size=32,num_layers=1,token_refiner_num_layers=1,num_attention_heads=2,attention_head_dim=16,ffn_hidden_size=64,text_dim=24,time_embed_hidden_size=32,time_embed_dim=16,timestep_input_dim=8,rope_inv_freq_len=4,dtype=dtype,device=torch.device("cuda"),operations=ops.manual_cast)
 with torch.no_grad():
  for name,p in model.named_parameters():
   p.fill_(1.0) if "norm" in name and p.ndim==1 else p.uniform_(-.02,.02)
 return model
try:
 fp32=make(torch.float32)
 x=torch.randn(8,32,device="cuda",dtype=torch.float32)
 t=torch.randn(1,16,device="cuda",dtype=torch.float32)
 seg=[(0,8,0)]
 with torch.inference_mode():
  baseline=fp32.blocks[0](x.clone(),t,seg,None)
  baseline_mlp=fp32.blocks[0].mlp(x.clone())
 spec=importlib.util.spec_from_file_location("isolated_h3_fp16_fix",ROOT/"minimax_h3_fp16_fix.py")
 patch=importlib.util.module_from_spec(spec);spec.loader.exec_module(patch)
 calls=[]
 def attention32(h,**kw):
  calls.append(str(h.dtype));return fp32.blocks[0].attn(h,**kw)
 with torch.inference_mode():
  after=fp32.blocks[0](x.clone(),t,seg,None,attention=attention32)
  after_mlp=fp32.blocks[0].mlp(x.clone())
 err=float((after-baseline).abs().max())
 ok=err==0 and torch.equal(baseline_mlp,after_mlp) and calls==["torch.float32"] and not getattr(fp32.blocks[0],"_h3_fp16_fix",False)
 results["tests"].append({"name":"fp32_unchanged_and_callback_forwarded","passed":ok,"max_abs_error":err,"callback_dtypes":calls})
 fp16=make(torch.float16)
 block=fp16.blocks[0]
 observed={"attention":[],"fc1":[],"fc2":[],"norm1":[],"norm2":[]}
 hooks=[]
 for key,module in [("fc1",block.mlp.fc1),("fc2",block.mlp.fc2),("norm1",block.norm1),("norm2",block.norm2)]:
  def record(mod,args,key=key):observed[key].append(str(args[0].dtype))
  hooks.append(module.register_forward_pre_hook(record))
 def attention16(h,**kw):
  observed["attention"].append(str(h.dtype));return block.attn(h,**kw)
 with torch.inference_mode():
  large=torch.randn(8,32,device="cuda",dtype=torch.float32)*100000
  out=block(large.clone(),t.to(torch.float16),seg,None,attention=attention16)
  default_out=block(large.clone(),t.to(torch.float16),seg,None)
  embeds=fp16.preprocess_text_embeds(torch.randn(1,8,24,device="cuda",dtype=torch.float16))
 torch.cuda.synchronize()
 for hook in hooks:hook.remove()
 finite=bool(torch.isfinite(out).all() and torch.isfinite(default_out).all() and torch.isfinite(embeds).all())
 branch_ok=observed["attention"]==["torch.float16"] and all(d=="torch.float16" for key in ["fc1","fc2"] for d in observed[key])
 residual_ok=all(d=="torch.float32" for key in ["norm1","norm2"] for d in observed[key])
 ok=finite and branch_ok and residual_ok and out.dtype==torch.float32 and getattr(block,"_h3_fp16_fix",False) and torch.equal(out,default_out)
 results["tests"].append({"name":"fp16_branches_fp32_residual_and_attention_override","passed":bool(ok),"finite":finite,"output_dtype":str(out.dtype),"text_projection_dtype":str(embeds.dtype),"input_abs_max":float(large.abs().max()),"observed_input_dtypes":observed,"callback_matches_default":bool(torch.equal(out,default_out))})
 results["peak_allocated_bytes"]=torch.cuda.max_memory_allocated()
 results["passed"]=all(r["passed"] for r in results["tests"])
except Exception as exc:
 results.update(passed=False,error=type(exc).__name__+": "+str(exc),traceback=traceback.format_exc())
print(json.dumps(results,indent=2))
sys.exit(0 if results["passed"] else 1)

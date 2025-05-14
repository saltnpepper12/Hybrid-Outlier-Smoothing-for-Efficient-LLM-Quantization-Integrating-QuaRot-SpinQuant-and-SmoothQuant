import torch
import copy
import pandas as pd
import numpy as np
from tqdm import tqdm

# Import QuaRot logic
import sys
sys.path.append('quarot2/QuaRot/fake_quant')
import rotation_utils as quarot_rotation
import gptq_utils as quarot_gptq
import quant_utils as quarot_quant

# Import SpinQuant logic
sys.path.append('SpinQuant/eval_utils')
import rotation_utils as spinquant_rotation
import gptq_utils as spinquant_gptq
import quant_utils as spinquant_quant

# Import SmoothQuant logic
sys.path.append('smoothquant/smoothquant')
import smooth as smoothquant_smooth
import quantize_llama_layer as smoothquant_quant  # adjust as needed

def analyze_layer(method, layer, args, calibration_batches):
    results = []
    layer_name = f"model.layers.{args.layer_idx}"
    # Clone the layer for quantization
    layer_q = copy.deepcopy(layer)
    if method == "QuaRot":
        # Apply QuaRot rotation
        config = args.model.config
        Q = quarot_rotation.get_orthogonal_matrix(config.hidden_size, args.rotate_mode, device=args.device)
        model_type = "llama"  # or use your extractor
        num_heads = config.num_attention_heads
        model_dim = config.hidden_size
        head_dim = model_dim // num_heads
        quarot_rotation.rotate_attention_inputs(layer_q, Q, model_type)
        quarot_rotation.rotate_attention_output(layer_q, Q, model_type)
        quarot_rotation.rotate_mlp_input(layer_q, Q, model_type)
        quarot_rotation.rotate_mlp_output(layer_q, Q, model_type)
        quarot_rotation.rotate_ov_proj(layer_q, model_type, num_heads, head_dim)
        gptq = quarot_gptq.GPTQ(layer_q)
        gptq.quantizer = quarot_quant.WeightQuantizer()
        gptq.quantizer.configure(args.w_bits, perchannel=True, sym=not(args.w_asym), mse=args.w_clip)
    elif method == "SpinQuant":
        config = args.model.config
        R1 = spinquant_rotation.get_orthogonal_matrix(config.hidden_size, args.rotate_mode, device=args.device)
        num_heads = config.num_attention_heads
        model_dim = config.hidden_size
        head_dim = model_dim // num_heads
        spinquant_rotation.rotate_attention_inputs(layer_q, R1)
        spinquant_rotation.rotate_attention_output(layer_q, R1)
        spinquant_rotation.rotate_mlp_input(layer_q, R1)
        spinquant_rotation.rotate_mlp_output(layer_q, R1)
        R2 = spinquant_rotation.get_orthogonal_matrix(head_dim, args.rotate_mode, device=args.device)
        spinquant_rotation.rotate_ov_proj(layer_q, num_heads, head_dim, R2=R2)
        gptq = spinquant_gptq.GPTQ(layer_q)
        gptq.quantizer = spinquant_quant.WeightQuantizer()
        gptq.quantizer.configure(args.w_bits, perchannel=True, sym=not(args.w_asym), mse=args.w_clip)
    elif method == "SmoothQuant":
        # Apply SmoothQuant logic (you may need to adapt this to your code)
        smoothquant_smooth.smooth_llama_layer(layer_q, act_scales=None, alpha=args.sq_alpha, layer_name=layer_name)
        smoothquant_quant.quantize_llama_layer(layer_q)
        # For SmoothQuant, you may not need GPTQ, just compare weights directly
    else:
        raise ValueError("Unknown method")

    # Collect input/output activations for calibration
    inps, outs = [], []
    def hook_fn(module, inp, out):
        inps.append(inp[0].detach().cpu())
        outs.append(out.detach().cpu())
    handle = layer_q.register_forward_hook(hook_fn)
    with torch.no_grad():
        for batch in calibration_batches:
            layer_q = layer_q.to(args.device)
            _ = layer_q(batch)
    handle.remove()
    if method in ["QuaRot", "SpinQuant"]:
        for inp, out in zip(inps, outs):
            gptq.add_batch(inp, out)
        gptq.fasterquant(percdamp=args.percdamp, groupsize=args.w_groupsize, actorder=args.act_order, static_groups=False)
    # Compare original and quantized weights for all submodules
    for name, mod in layer.named_modules():
        if isinstance(mod, torch.nn.Linear):
            orig_weight = mod.weight.data.cpu().float().flatten().numpy()
            quant_weight = getattr(layer_q, name).weight.data.cpu().float().flatten().numpy() if hasattr(layer_q, name) else None
            if quant_weight is None:
                continue
            norm_orig = np.linalg.norm(orig_weight)
            norm_quant = np.linalg.norm(quant_weight)
            cos_sim = 0.0 if norm_orig == 0 or norm_quant == 0 else float(np.clip(np.dot(orig_weight, quant_weight) / (norm_orig * norm_quant), -1.0, 1.0))
            mse = float(np.mean((orig_weight - quant_weight) ** 2))
            results.append({
                'layer_idx': args.layer_idx,
                'layer_name': layer_name,
                'submodule': name,
                'cosine_similarity': cos_sim,
                'mse': mse,
                'method': method
            })
    return results

def main():
    # Parse args, load model, tokenizer, calibration data, etc.
    # For each layer:
    #   for method in ["QuaRot", "SpinQuant", "SmoothQuant"]:
    #       results += analyze_layer(method, layer, args, calibration_batches)
    # Save/print results as CSV/DataFrame
    pass

if __name__ == "__main__":
    main()

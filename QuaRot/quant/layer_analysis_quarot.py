import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from tqdm import tqdm
import model_utils
import data_utils
import quant_utils
import gptq_utils
import utils
import copy
import rotation_utils

class QuaRotLayerAnalyzer:
    def __init__(self, model_name, args, device="cuda"):
        self.device = device
        self.args = args
        self.model, self.tokenizer = model_utils.get_model_and_tokenizer(model_name, args.hf_token)
        self.model.eval()
        self.model.to(self.device)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        self.seqlen = self.model.seqlen

        # Load dataset
        self.dataset = data_utils.get_loaders(
            args.cal_dataset, nsamples=args.nsamples,
            seed=args.seed, model=args.model,
            seqlen=self.seqlen, eval_mode=False
        )

    def get_calibration_data(self, n_samples=5, seq_length=128):
        # Use the first n_samples batches from the dataloader
        batches = []
        for i, batch in enumerate(self.dataset):
            if i >= n_samples:
                break
            batches.append(batch[0].to(self.device))
        return batches

    def quantize_layer(self, layer, args):
        # Use GPTQ quantization logic for a single layer
        gptq = gptq_utils.GPTQ(layer)
        gptq.quantizer = quant_utils.WeightQuantizer()
        gptq.quantizer.configure(
            args.w_bits, perchannel=True, sym=not(args.w_asym), mse=args.w_clip
        )
        return gptq

    def analyze_single_layer(self, layer_idx, batches, args):
        layer = self.model.model.layers[layer_idx]
        layer_name = f"model.layers.{layer_idx}"
        results = []
        # Clone the layer for quantization
        layer_q = copy.deepcopy(layer)
        # --- QuaRot rotation logic ---
        config = self.model.config
        Q = rotation_utils.get_orthogonal_matrix(config.hidden_size, args.rotate_mode, device=self.device)
        model_type = model_utils.model_type_extractor(self.model)
        num_heads = config.num_attention_heads
        model_dim = config.hidden_size
        head_dim = model_dim // num_heads
        rotation_utils.rotate_attention_inputs(layer_q, Q, model_type)
        rotation_utils.rotate_attention_output(layer_q, Q, model_type)
        rotation_utils.rotate_mlp_input(layer_q, Q, model_type)
        rotation_utils.rotate_mlp_output(layer_q, Q, model_type)
        rotation_utils.rotate_ov_proj(layer_q, model_type, num_heads, head_dim)
        # --- End QuaRot rotation logic ---
        gptq = self.quantize_layer(layer_q, args)
        # Collect input/output activations for calibration
        inps = []
        outs = []
        def hook_fn(module, inp, out):
            inps.append(inp[0].detach().cpu())
            outs.append(out.detach().cpu())
        handle = layer_q.register_forward_hook(hook_fn)
        with torch.no_grad():
            for batch in batches:
                self.model.model.embed_tokens = self.model.model.embed_tokens.to(self.device)
                self.model.model.norm = self.model.model.norm.to(self.device)
                layer_q = layer_q.to(self.device)
                # Forward pass through the layer
                _ = layer_q(batch)
        handle.remove()
        # Add collected batches to GPTQ
        for inp, out in zip(inps, outs):
            gptq.add_batch(inp, out)
        # Quantize weights
        gptq.fasterquant(
            percdamp=args.percdamp, groupsize=args.w_groupsize, actorder=args.act_order, static_groups=False
        )
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
                    'layer_idx': layer_idx,
                    'layer_name': layer_name,
                    'submodule': name,
                    'cosine_similarity': cos_sim,
                    'mse': mse
                })
        return results

    def analyze_all_layers(self, n_samples=5, seq_length=128):
        batches = self.get_calibration_data(n_samples=n_samples, seq_length=seq_length)
        all_results = []
        for layer_idx in tqdm(range(len(self.model.model.layers)), desc="Analyzing QuaRot layers"):
            layer_results = self.analyze_single_layer(layer_idx, batches, self.args)
            all_results.extend(layer_results)
        df = pd.DataFrame(all_results)
        print("\nQuaRot Layer-wise Weight Analysis:")
        print(df.to_string())
        df.to_csv('quarot_layer_weight_analysis.csv', index=False)
        print("\nResults saved to 'quarot_layer_weight_analysis.csv'")
        return df

if __name__ == "__main__":
    import utils
    args = utils.parser_gen()
    analyzer = QuaRotLayerAnalyzer(args.model, args)
    results_df = analyzer.analyze_all_layers(n_samples=5, seq_length=128)
    print("\nLayer Analysis Summary:")
    print("======================")
    poor_layers = results_df[results_df['cosine_similarity'] < 0.95]
    print("\nLayers with Poor Performance:")
    print("============================")
    print(poor_layers[['layer_idx', 'layer_name', 'submodule', 'cosine_similarity', 'mse']].to_string())
    print("\nPerformance by Layer Type:")
    print("========================")
    type_stats = results_df.groupby('submodule').agg({
        'cosine_similarity': ['mean', 'std', 'min', 'max'],
        'mse': ['mean', 'std', 'min', 'max']
    })
    print(type_stats)
    results_df.to_csv('quarot_layer_analysis_results.csv', index=False)
    print("\nResults saved to 'quarot_layer_analysis_results.csv'") 
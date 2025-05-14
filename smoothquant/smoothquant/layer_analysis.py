import torch
import torch.nn.functional as F
from transformers import LlamaForCausalLM, LlamaTokenizer
from datasets import load_dataset
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from smooth import smooth_lm, smooth_linear_weight, smooth_llama_layer
from fake_quant import quantize_llama_like, quantize_linear_weight, quantize_llama_layer
from huggingface_hub import login
import os
import math
import torch.nn as nn
import numpy as np

def setup_hf_auth():
    """Setup Hugging Face authentication"""
    if "HUGGINGFACE_TOKEN" in os.environ:
        token = os.environ["HUGGINGFACE_TOKEN"]
    else:
        token = input("Please enter your Hugging Face token: ")
    login(token=token)
    print("Successfully logged in to Hugging Face!")

class LayerAnalyzer:
    def __init__(self, model_name="meta-llama/Llama-2-7b-hf", device="cuda"):
        self.device = device
        self.tokenizer = LlamaTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        
        # Load original model
        self.model_fp16 = LlamaForCausalLM.from_pretrained(
            model_name, 
            torch_dtype=torch.float16, 
            device_map="auto",
            pad_token_id=self.tokenizer.pad_token_id
        )
        self.model_fp16.eval()
        
        # Load dataset
        self.dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
        
        # Define layer names
        self.layer_names = {
            'embed_tokens': 'Input Embeddings',
            'norm': 'Input Layer Norm',
            'layers': ['Transformer Layer {}'.format(i) for i in range(len(self.model_fp16.model.layers))],
            'lm_head': 'Output Layer'
        }

    def get_calibration_data(self, n_samples=5, seq_length=128):
        """Get calibration data for analysis"""
        texts = self.dataset["text"][:n_samples]
        inputs = self.tokenizer(
            texts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=seq_length
        ).to(self.device)
        return inputs

    def get_rotary_embeddings(self, seq_length, hidden_size, num_heads):
        """Generate rotary position embeddings"""
        # Calculate the dimension of each head
        head_dim = hidden_size // num_heads
        
        # Generate the inverse frequency
        inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2).float().to(self.device) / head_dim))
        
        # Generate the position indices
        t = torch.arange(seq_length, device=self.device).type_as(inv_freq)
        
        # Calculate the frequencies
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        
        # Generate the embeddings
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()
        sin = emb.sin()
        
        # Reshape for broadcasting
        cos = cos.view(seq_length, 1, 1, head_dim)
        sin = sin.view(seq_length, 1, 1, head_dim)
        
        return cos, sin

    def quantize_layer(self, layer):
        """Quantize a single layer's weights"""
        with torch.no_grad():
            # Quantize attention weights
            if hasattr(layer, 'self_attn'):
                layer.self_attn.q_proj.weight.data = torch.fake_quantize_per_tensor_affine(
                    layer.self_attn.q_proj.weight.data, 1.0, 0, -128, 127
                )
                layer.self_attn.k_proj.weight.data = torch.fake_quantize_per_tensor_affine(
                    layer.self_attn.k_proj.weight.data, 1.0, 0, -128, 127
                )
                layer.self_attn.v_proj.weight.data = torch.fake_quantize_per_tensor_affine(
                    layer.self_attn.v_proj.weight.data, 1.0, 0, -128, 127
                )
                layer.self_attn.o_proj.weight.data = torch.fake_quantize_per_tensor_affine(
                    layer.self_attn.o_proj.weight.data, 1.0, 0, -128, 127
                )
            
            # Quantize MLP weights
            if hasattr(layer, 'mlp'):
                layer.mlp.gate_proj.weight.data = torch.fake_quantize_per_tensor_affine(
                    layer.mlp.gate_proj.weight.data, 1.0, 0, -128, 127
                )
                layer.mlp.up_proj.weight.data = torch.fake_quantize_per_tensor_affine(
                    layer.mlp.up_proj.weight.data, 1.0, 0, -128, 127
                )
                layer.mlp.down_proj.weight.data = torch.fake_quantize_per_tensor_affine(
                    layer.mlp.down_proj.weight.data, 1.0, 0, -128, 127
                )
            
            return layer

    def analyze_single_layer(self, layer_idx, inputs, act_scales, alpha=0.85):
        """Layer-wise analysis: cosine similarity and MSE for each submodule weight (Attention and MLP), using true SmoothQuant/quantization logic."""
        import copy
        from transformers.models.llama.modeling_llama import LlamaDecoderLayer
        layer = self.model_fp16.model.layers[layer_idx]
        layer_name = f"model.layers.{layer_idx}"
        results = []
        with torch.no_grad():
            # Clone the layer for quantization
            layer_sq = copy.deepcopy(layer)
            # Apply SmoothQuant using the real logic
            smooth_llama_layer(layer_sq, act_scales, alpha, layer_name=layer_name)
            # Quantize using the real logic
            quantize_llama_layer(layer_sq)
            # Compare weights for all submodules
            # Attention
            for proj_name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
                orig_proj = getattr(layer.self_attn, proj_name, None)
                quant_proj = getattr(layer_sq.self_attn, proj_name, None)
                if orig_proj is not None and quant_proj is not None:
                    orig_weight = orig_proj.weight.data.cpu().float().flatten().numpy()
                    quant_weight = quant_proj.weight.data.cpu().float().flatten().numpy()
                    # Robust cosine similarity calculation
                    norm_orig = np.linalg.norm(orig_weight)
                    norm_quant = np.linalg.norm(quant_weight)
                    if norm_orig == 0 or norm_quant == 0:
                        cos_sim = 0.0
                    else:
                        cos_sim = np.dot(orig_weight, quant_weight) / (norm_orig * norm_quant)
                        cos_sim = float(np.clip(cos_sim, -1.0, 1.0))
                    mse = float(np.mean((orig_weight - quant_weight) ** 2))
                    results.append({
                        'layer_idx': layer_idx,
                        'layer_name': self.layer_names['layers'][layer_idx],
                        'submodule': f'self_attn.{proj_name}',
                        'cosine_similarity': cos_sim,
                        'mse': mse
                    })
            # MLP
            for proj_name in ['gate_proj', 'up_proj', 'down_proj']:
                orig_proj = getattr(layer.mlp, proj_name, None)
                quant_proj = getattr(layer_sq.mlp, proj_name, None)
                if orig_proj is not None and quant_proj is not None:
                    orig_weight = orig_proj.weight.data.cpu().float().flatten().numpy()
                    quant_weight = quant_proj.weight.data.cpu().float().flatten().numpy()
                    norm_orig = np.linalg.norm(orig_weight)
                    norm_quant = np.linalg.norm(quant_weight)
                    if norm_orig == 0 or norm_quant == 0:
                        cos_sim = 0.0
                    else:
                        cos_sim = np.dot(orig_weight, quant_weight) / (norm_orig * norm_quant)
                        cos_sim = float(np.clip(cos_sim, -1.0, 1.0))
                    mse = float(np.mean((orig_weight - quant_weight) ** 2))
                    results.append({
                        'layer_idx': layer_idx,
                        'layer_name': self.layer_names['layers'][layer_idx],
                        'submodule': f'mlp.{proj_name}',
                        'cosine_similarity': cos_sim,
                        'mse': mse
                    })
        return results

    def analyze_all_layers(self, act_scales_path, n_samples=5, seq_length=128):
        act_scales = torch.load(act_scales_path)
        all_results = []
        for layer_idx in tqdm(range(len(self.model_fp16.model.layers)), desc="Analyzing layers"):
            layer_results = self.analyze_single_layer(layer_idx, None, act_scales)
            all_results.extend(layer_results)
        df = pd.DataFrame(all_results)
        print("\nLayer-wise Weight Analysis:")
        print(df.to_string())
        df.to_csv('layer_weight_analysis.csv', index=False)
        print("\nResults saved to 'layer_weight_analysis.csv'")
        return df

def main():
    # Setup Hugging Face authentication
    setup_hf_auth()
    
    # Initialize analyzer
    analyzer = LayerAnalyzer()
    
    # Run analysis
    results_df = analyzer.analyze_all_layers(
        act_scales_path="act_scales/llama-2-7b.pt",
        n_samples=5,
        seq_length=128
    )
    
    # Print summary
    print("\nLayer Analysis Summary:")
    print("======================")
    
    # Print layers with poor performance
    poor_layers = results_df[results_df['cosine_similarity'] < 0.95]
    print("\nLayers with Poor Performance:")
    print("============================")
    print(poor_layers[['layer_idx', 'layer_name', 'submodule', 'cosine_similarity', 'mse']].to_string())
    
    # Print layer type statistics
    print("\nPerformance by Layer Type:")
    print("========================")
    type_stats = results_df.groupby('submodule').agg({
        'cosine_similarity': ['mean', 'std', 'min', 'max'],
        'mse': ['mean', 'std', 'min', 'max']
    })
    print(type_stats)
    
    # Save results to CSV
    results_df.to_csv('layer_analysis_results.csv', index=False)
    print("\nResults saved to 'layer_analysis_results.csv'")

if __name__ == "__main__":
    main()
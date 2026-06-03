# Hybrid Outlier Smoothing for Efficient LLM Quantization

A unified, **layer-wise** framework for analyzing and combining three state-of-the-art
LLM quantization methods — **QuaRot**, **SpinQuant**, and **SmoothQuant** — to guide
mixed-precision (INT4/INT8) assignment per transformer submodule.

The core idea: not every layer tolerates aggressive quantization equally. This framework
measures, per submodule, how faithfully each method reconstructs the original FP16 weights
(cosine similarity and MSE), then assigns the right method and precision to each layer
instead of applying one technique uniformly.

> **Methods compared**
> - **QuaRot** — quantization with rotation (Hadamard or random orthogonal)
> - **SpinQuant** — quantization with optimized or random rotations
> - **SmoothQuant** — quantization with activation-outlier smoothing

---

## How it works

The pipeline runs in two stages.

**1. Per-method, per-layer diagnostics.** For each method, for every transformer submodule:
1. Clone the layer.
2. Apply the method's rotation/smoothing logic (if any).
3. Quantize the processed weights.
4. Compare original vs. processed+quantized weights — cosine similarity and MSE.
5. Aggregate into a CSV and flag poorly quantized submodules (cosine < 0.95).

Calibration uses a small number of Wikitext-2 batches for realistic activation statistics.
Each method writes one CSV; the three are merged into
[`Outputs/CSV-FILES/merged_layerwise_analysis.csv`](Outputs/CSV-FILES/merged_layerwise_analysis.csv).

**2. Hybrid assignment.** [`hybrid.py`](hybrid.py) consumes the merged diagnostics and, per
submodule, (a) picks the method with the highest reconstruction fidelity and (b) assigns a
precision tier — submodules that quantize cleanly go to INT4, the more sensitive ones are
kept at INT8. It emits a mixed-precision assignment plan plus a summary of the method mix,
precision mix, and average bit-width / compression vs. FP16. This stage needs no GPU.

---

## What's in this repo

| Path | What it is |
|------|------------|
| [`hybrid.py`](hybrid.py) | **My contribution** — hybrid assignment engine (per-submodule method + INT4/INT8 selection). |
| [`smoothquant/smoothquant/layer_analysis.py`](smoothquant/smoothquant/layer_analysis.py) | **My contribution** — per-submodule cosine/MSE diagnostics for SmoothQuant. |
| [`QuaRot/quant/layer_analysis_quarot.py`](QuaRot/quant/layer_analysis_quarot.py) | **My contribution** — per-submodule diagnostics for QuaRot. |
| [`SpinQuant/eval_utils/layer_analysis_spinquant.py`](SpinQuant/eval_utils/layer_analysis_spinquant.py) | **My contribution** — per-submodule diagnostics for SpinQuant. |
| `QuaRot/`, `SpinQuant/`, `smoothquant/` (everything else) | Upstream reference implementations this work builds on (see [Acknowledgements](#acknowledgements)). |
| [`Outputs/`](Outputs/) | Generated CSVs and result plots. |

The `layer_analysis*` scripts and `hybrid.py` are what I wrote; the surrounding method
directories are the official upstream implementations the analysis runs on top of.

---

## Quick start

```bash
git clone https://github.com/saltnpepper12/Hybrid-Outlier-Smoothing-for-Efficient-LLM-Quantization-Integrating-QuaRot-SpinQuant-and-SmoothQuant.git
cd Hybrid-Outlier-Smoothing-for-Efficient-LLM-Quantization-Integrating-QuaRot-SpinQuant-and-SmoothQuant

conda create -n quant-analysis python=3.10 -y
conda activate quant-analysis
pip install -r requirements.txt
```

`meta-llama/Llama-2-7b-hf` is gated; authenticate with `huggingface-cli login` (or set the
`HUGGINGFACE_TOKEN` environment variable) before running the analyzers.

### Run the layer-wise diagnostics

```bash
# SmoothQuant
python smoothquant/smoothquant/layer_analysis.py --model meta-llama/Llama-2-7b-hf

# QuaRot
python QuaRot/quant/layer_analysis_quarot.py --model meta-llama/Llama-2-7b-hf --rotate_mode hadamard

# SpinQuant
python SpinQuant/eval_utils/layer_analysis_spinquant.py --model meta-llama/Llama-2-7b-hf
```

**Common args:** `--model` (HF name or path), `--w_bits` (default 4), `--rotate_mode`
(`hadamard` | `random`), `--nsamples` (calibration samples), `--seed`.
See each script's `--help` for the full list.

### Build the hybrid assignment

Once the per-method diagnostics are merged into the merged CSV, the assignment stage runs
anywhere — no GPU or model download required:

```bash
python hybrid.py \
  --input Outputs/CSV-FILES/merged_layerwise_analysis.csv \
  --output Outputs/hybrid_assignment.csv \
  --int4-threshold 0.95
```

This writes `Outputs/hybrid_assignment.csv` (the per-submodule method + precision plan) and
prints the method mix, precision mix, and average bit-width / compression vs. FP16.

---

## Results

The primary results are **per-submodule reconstruction diagnostics** (cosine similarity and
MSE between the original FP16 weights and each method's quantized weights), across all
transformer layers of Llama-2-7B:

| Artifact | Contents |
|----------|----------|
| [`Outputs/CSV-FILES/SmoothQuant.csv`](Outputs/CSV-FILES/) | Per-submodule cosine/MSE for SmoothQuant |
| [`Outputs/CSV-FILES/QuaRot.csv`](Outputs/CSV-FILES/) | Per-submodule cosine/MSE for QuaRot |
| [`Outputs/CSV-FILES/SpinQuant.csv`](Outputs/CSV-FILES/) | Per-submodule cosine/MSE for SpinQuant |
| [`Outputs/CSV-FILES/merged_layerwise_analysis.csv`](Outputs/CSV-FILES/merged_layerwise_analysis.csv) | All three methods side by side, per submodule |
| [`Outputs/Results-Images/`](Outputs/Results-Images/) | Benchmark and per-layer comparison plots |

Method-vs-method comparison plot:

![Comparison](Outputs/Results-Images/Comparison.png)

Key observations from the diagnostics:
- Output/down projections (`o_proj`, `down_proj`) quantize almost losslessly (cosine ≈ 0.99+)
  under all three methods — strong INT4 candidates.
- Query/key/value projections in early layers are the most sensitive (lower cosine), and are
  the submodules `hybrid.py` keeps at INT8.

*Hardware: single NVIDIA A100 (40GB+). Calibration: Wikitext-2.*

---

## Hardware

The per-method diagnostics were run on a single NVIDIA A100. For 7B models and above, use a
GPU with ≥40GB to avoid OOM. The `hybrid.py` assignment stage runs on CPU.

---

## Acknowledgements

Builds on the official implementations of
[QuaRot](https://github.com/spcl/QuaRot),
[SpinQuant](https://github.com/facebookresearch/SpinQuant), and
[SmoothQuant](https://github.com/mit-han-lab/smoothquant). If you use this framework, please
also cite the original papers for each method.

## License

Released under the [MIT License](LICENSE).

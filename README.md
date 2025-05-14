# Hybrid Outlier Smoothing for Efficient LLM Quantization: Integrating QuaRot, SpinQuant, and SmoothQuant

This project provides a unified framework for **layer-wise analysis of quantization techniques** for Large Language Models (LLMs), focusing on three state-of-the-art methods:

- **QuaRot**: Quantization with Rotation (Hadamard or random orthogonal)
- **SpinQuant**: Quantization with optimized or random rotations
- **SmoothQuant**: Quantization with outlier smoothing

Our goal is to enable **side-by-side, per-layer analysis** of how each method affects model weights, using robust metrics (cosine similarity, MSE) to guide research and practical deployment.

---

## Project Structure & Methodology

- **Layer-wise Analysis**: For each method, we:
  1. Clone each transformer layer.
  2. Apply the method's rotation/smoothing logic (if any).
  3. Quantize the processed weights.
  4. Compare original vs processed+quantized weights (cosine similarity, MSE).
  5. Aggregate results into a CSV and print a summary.
- **Calibration Data**: Uses a small number of batches from Wikitext-2 (or other datasets) for realistic activation statistics.
- **Metrics**: Cosine similarity and mean squared error (MSE) between original and quantized weights, per submodule.

---

## Environment Setup

We recommend using **conda** or **virtualenv** for isolation. Each method may have slightly different requirements (see below).

### 1. Clone the repository and submodules
```bash
git clone <your-repo-url>
cd <repo-root>
# If using submodules for SpinQuant, QuaRot, etc.:
git submodule update --init --recursive
```

### 2. Create and activate environment
```bash
conda create -n quant-analysis python=3.10 -y
conda activate quant-analysis
# or use virtualenv if you prefer
```

### 3. Install dependencies
- For each method, install its requirements (see below for details).

---

## Running Layer-wise Analysis

### **A. SmoothQuant**

**Directory:** `smoothquant/smoothquant/`

**Setup:**
```bash
cd smoothquant
pip install -r requirements.txt
```

**Run layer-wise analysis:**
```bash
python smoothquant/layer_analysis.py --model <model_name_or_path> [other args]
```
- Results: CSV and console output in the same directory.

---

### **B. QuaRot**

**Directory:** `QuaRot/quant/`

**Setup:**
```bash
cd QuaRot/quant
pip install -r ../../requirements.txt  # or the appropriate requirements file
```

**Run layer-wise analysis:**
```bash
python layer_analysis_quarot.py --model <model_name_or_path> [other args]
```
- Results: CSV and console output in the same directory.

---

### **C. SpinQuant**

**Directory:** `SpinQuant/eval_utils/`

**Setup:**
```bash
cd SpinQuant
pip install -r requirement.txt
```

**Run layer-wise analysis:**
```bash
python eval_utils/layer_analysis_spinquant.py --model <model_name_or_path> [other args]
```
- Results: CSV and console output in the same directory.

---

## Example Arguments
- `--model`: HuggingFace model name or local path (e.g., `meta-llama/Llama-2-7b-hf`)
- `--w_bits`: Number of bits for weight quantization (default: 4)
- `--rotate_mode`: Rotation type (`hadamard` or `random`)
- `--nsamples`: Number of calibration samples (default: 5)
- `--seed`: Random seed

See each script's `--help` for full options.

---

## Results & Interpretation
- Each script outputs a CSV with per-layer, per-submodule cosine similarity and MSE.
- Poorly quantized layers (e.g., cosine similarity < 0.95) are highlighted in the console.
- Use these results to compare methods, tune quantization parameters, or guide further research.

---

## Hardware Used
- All experiments and analyses in this project were run on a single NVIDIA A100 GPU.
- For best results and to avoid out-of-memory errors, we recommend using a GPU with at least 40GB of memory (such as the A100) for large models like Llama-2-7B and above.

---

## Troubleshooting & Tips
- Ensure you have enough GPU memory for the chosen model and batch size.
- If you encounter missing dependencies, check the requirements file for each method.
- For custom datasets or models, adapt the calibration data loader as needed.
- If you want to compare with/without rotation, comment out the rotation logic in the relevant script.

---

## Contributing & Contact
- Pull requests and issues are welcome!
- For questions, please open an issue or contact the maintainer.

---

## Citation
If you use this framework in your research, please cite the original papers for QuaRot, SpinQuant, and SmoothQuant, as well as this repository.

---

## Acknowledgements
- This project builds on the official implementations of QuaRot, SpinQuant, and SmoothQuant.
- Thanks to the open-source community for making this research possible. 

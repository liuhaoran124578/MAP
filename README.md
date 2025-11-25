# 🏆 MAP: Charting Student Math Misunderstandings (Kaggle)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Experiment Tracking](https://img.shields.io/badge/SwanLab-Enabled-4db6ac)](https://swanlab.cn)

This repository contains the reproduction and implementation of top solutions for the Kaggle competition **[MAP - Charting Student Math Misunderstandings](https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics)**.

Currently, we have successfully reproduced the **1st Place Solution** and plan to integrate other top strategies.

## 📰 News

- **[2025-11-25]** ⚡ **W8A8 Quantization Support**: Added SmoothQuant implementation with custom Triton kernels. You can now quantize 32B+ models to INT8 for efficient inference on limited VRAM.
- **[2025-11-25]** 🚀 **1st Place Solution (tascj)** reproduction is now fully supported!
    - Features: Qwen2.5/3 & GLM-4 backbones, OffloadAdam optimizer, and robust evaluation (MAP@3).
    - Integrated **SwanLab** for experiment tracking.

## 📊 Solution Summary

We aim to implement the following top 10 solutions. Detailed analysis can be found in [Summary.md](./Summary.md).

| Rank | Team / Solution | Status | Backbone Models | Key Features |
| :---: | :--- | :---: | :--- | :--- |
| 🥇 | **1st Place (tascj)** | ✅ Done | Qwen2.5, Qwen3, GLM4 | Suffix Classification, Stochastic Rounding, **W8A8 SmoothQuant** |
| 🥉 | 3rd Place | 🚧 TODO | - | - |
| 4️⃣ | 4th Place | 🚧 TODO | - | - |
| 🔟 | 10th Place | 🚧 TODO | - | - |
| - | ... | 🚧 TODO | - | - |

## 📂 Project Structure

The project is organized as follows:

```text
.
├── config                 # Configuration files (YAML)
│   └── tascj              # Configs for 1st place solution
│       ├── x_0.yaml       # Experiment Config
│       └── ...
├── data                   # Dataset directory
├── tascj                  # 1st Place Solution Codebase
│   ├── scripts            # Shell scripts
│   │   ├── convert.sh     # Script for W8A8 quantization
│   │   └── train.sh       # Script for training
│   └── src                # Source code
│       ├── config.py      # Pydantic configuration schemas
│       ├── dataset.py     # MAPDataset & Data Collator
│       ├── train.py       # Main training loop
│       ├── inference      # Inference & Quantization tools
│       │   ├── map-submit.ipynb # Submission notebook
│       │   ├── sq_collect.py    # Step 1: Collect activation scales
│       │   └── sq_convert.py    # Step 2: Convert weights to W8A8
│       └── modules        # Core modules
│           ├── models
│           │   ├── w8a8_kernels.py # Custom Triton kernels
│           │   └── ...
│           └── optim      # Custom Optimizers (OffloadAdam)
├── requirements.txt       # Python dependencies
├── setup.sh               # Environment setup script
└── README.md
```

## 🚀 Quick Start

### 1. Environment Setup

We provide a convenient setup script to install all necessary dependencies (including `triton` for W8A8 kernels).

```bash
# Clone the repository
git clone git@github.com:liuhaoran124578/MAP.git
cd MAP
# Run setup script
bash setup.sh
```

### 2. Data Preparation

Please download the competition data from [Kaggle](https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/data) and place it in the `data/` directory.

### 3. Training

You can launch the training using the provided shell script.

```bash
# Syntax: bash tascj/scripts/train.sh <Config_Path> <Output_Dir>
bash tascj/scripts/train.sh config/tascj/x_0.yaml artifacts
```

**Key Config Arguments:**
- `llm_config.backbone`: Model path (e.g., `Qwen/Qwen2.5-32B`).
- `optimizer_config.name`: Use `OffloadAdam` to save GPU memory.

### 4. ⚡ W8A8 Quantization (SmoothQuant)

To deploy large models (e.g., Qwen 32B) on standard GPUs, we support **SmoothQuant (W8A8)**. This process converts both weights and activations to 8-bit integers.

**The process consists of two steps:**
1.  **Collect Scales**: Run inference on a calibration set to determine activation ranges.
2.  **Convert Weights**: Transform FP16/BF16 weights to INT8 based on the collected scales.

We provide a one-click script to handle this:

```bash
# Usage: bash tascj/scripts/convert.sh <Config_Path> <Checkpoint_Dir>

# Example:
bash tascj/scripts/convert.sh config/tascj/x_0.yaml artifacts/x_0/checkpoint_epoch_1
```

> **Note:** The quantized model will be saved in the `checkpoint_w8a8` folder inside your checkpoint directory.

### 5. Inference & Submission

For Kaggle submission or local inference using the quantized model:

1.  Ensure you have run the **W8A8 Quantization** step above.
2.  Use the `tascj/src/inference/map-submit.ipynb` notebook.
3.  Configure the notebook to load the `checkpoint_w8a8` model path.

## 🛠️ Implementation Details

- **W8A8 SmoothQuant**: Implemented using **OpenAI Triton**. We use custom kernels (`w8a8_kernels.py`) for:
    - Fused RMS Norm + Quantization.
    - INT8 Matrix Multiplication with dynamic per-token quantization.
- **OffloadAdam**: A custom optimizer that offloads optimizer states to CPU RAM, allowing 32B models to fine-tune on 24GB/40GB VRAM GPUs.
- **Suffix Classification**: A specific prompting strategy that frames the task as a next-token prediction problem for candidate suffixes.

## 🙏 Acknowledgements

We would like to thank the Kaggle community and the competition organizers. Special thanks to the authors of the following solutions and libraries:

- **1st Place Solution**: [tascj](https://github.com/tascj/kaggle-map-charting-student-math-misunderstandings) (Original implementation reference).
- **Transformers**: Hugging Face.
- **SwanLab**: For the modern experiment tracking tool.
- **Triton**: For high-performance GPU programming.

## 📝 License

This project is licensed under the MIT License. See the `LICENSE` file for details.
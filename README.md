
# 🏆 MAP: Charting Student Math Misunderstandings (Kaggle)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Experiment Tracking](https://img.shields.io/badge/SwanLab-Enabled-4db6ac)](https://swanlab.cn)

This repository contains the reproduction and implementation of top solutions for the Kaggle competition **[MAP - Charting Student Math Misunderstandings](https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics)**.

Currently, we have successfully reproduced the **1st Place Solution** and plan to integrate other top strategies.

## 📰 News

- **[2025-11-25]** 🚀 **1st Place Solution (tascj)** reproduction is now fully supported!
    - Features: Qwen2.5/3 & GLM-4 backbones, OffloadAdam optimizer, and robust evaluation (MAP@3).
    - Integrated **SwanLab** for experiment tracking.

## 📊 Solution Summary

We aim to implement the following top 10 solutions. Detailed analysis can be found in [Summary.md](./Summary.md).

| Rank | Team / Solution | Status | Backbone Models | Key Features |
| :---: | :--- | :---: | :--- | :--- |
| 🥇 | **1st Place (tascj)** | ✅ Done | Qwen2.5, Qwen3, GLM4 | Suffix Classification, Stochastic Rounding, OffloadAdam |
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
│       ├── x_1.yaml
│       └── ...
├── data                   # Dataset directory
│   ├── map-charting-student-math-misunderstandings.zip
│   ├── tascj              # Preprocessed data for tascj
│   ├── train.csv
│   └── ...
├── tascj                  # 1st Place Solution Codebase
│   ├── scripts            # Training/Inference scripts
│   └── src                # Source code
│       ├── config.py      # Pydantic configuration schemas
│       ├── dataset.py     # MAPDataset & Data Collator
│       ├── model.py       # Model wrapper (Qwen/GLM)
│       ├── optimizer.py   # Optimizer factory
│       ├── train.py       # Main training loop
│       └── modules        # Core modules (Optimization, Modeling)
├── requirements.txt       # Python dependencies
├── setup.sh               # Environment setup script
├── train.sh               # Entry point script (optional wrapper)
└── README.md
```

## 🚀 Quick Start

### 1. Environment Setup

We provide a convenient setup script to install all necessary dependencies.

```bash
# Clone the repository
git clone git@github.com:liuhaoran124578/MAP.git
cd MAP
# Run setup script
bash setup.sh
```

### 2. Data Preparation

Please download the competition data from [Kaggle](https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/data) and place it in the `data/` directory.

Ensure your `data/` folder looks like this:
```text
data/
├── train.csv
├── test.csv
├── sample_submission.csv
└── tascj/
    └── dtrainval.csv  # Processed dataset if applicable
```

### 3. Training

You can launch the training using the provided shell script. The script automatically handles path configurations.

**Usage:**
```bash
# Syntax: bash tascj/scripts/train.sh <Config_Path> <Output_Dir>
bash tascj/scripts/train.sh config/tascj/x_2.yaml artifacts
```

**Key Arguments in Config (`config/tascj/*.yaml`):**
- `exp_name`: Name of the experiment (affects output folder name).
- `llm_config.backbone`: Path or name of the HF model (e.g., `Qwen/Qwen2.5-32B-Instruct`).
- `optimizer_config.name`: Supports `OffloadAdam` for memory efficiency.
- `gradient_checkpointing`: Set to `true` to save VRAM.

### 4. Evaluation

Evaluation is performed automatically during training based on `eval_interval`. Results (MAP@3 score and Loss) are logged to the console and SwanLab.

To run evaluation only:
```bash
python tascj/src/train.py --config config/tascj/x_0.yaml --eval-only --load-from artifacts/your_exp_name/checkpoint_epoch_1
```

## 📈 Experiment Tracking

This project uses **SwanLab** for lightweight experiment tracking and visualization.
When you run the training script, a link to the experiment dashboard will be printed in the terminal.

Log files and artifacts are saved in: `artifacts/<exp_name>/`

## 🛠️ Implementation Details

- **Configuration Management**: Uses `Pydantic` for strict type checking of YAML configs.
- **Custom Optimizer**: Implements `OffloadAdam` with stochastic rounding to enable training large models (e.g., 32B) on consumer/cloud GPUs.
- **Data Pipeline**: A custom `MAPDataset` handles the "Suffix Classification" prompting strategy used by the 1st place solution.

## 🙏 Acknowledgements

We would like to thank the Kaggle community and the competition organizers. Special thanks to the authors of the following solutions and libraries:

- **1st Place Solution**: [tascj](https://github.com/tascj/kaggle-map-charting-student-math-misunderstandings) (Original implementation reference).
- **Transformers**: Hugging Face.
- **SwanLab**: For the modern experiment tracking tool.
- **Qwen & GLM**: For the powerful open-source LLMs.

## 📝 License

This project is licensed under the MIT License. See the `LICENSE` file for details.

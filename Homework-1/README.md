# Visual Recognition using Deep Learning - Homework 1
**Student ID:** 109550202
**Name:** 白詩愷

## Introduction
This project implements an Image Classification pipeline using **ResNet50**. To improve robustness and accuracy, I used a **Model Ensemble** approach combining three different training runs:
- **ResNet50 (Seed 0):** Optimized with Adam.
- **ResNet50 (Seed 42):** Optimized with Adam to test weight initialization variance.
- **ResNet50 (SGD):** Optimized using Stochastic Gradient Descent with momentum.

The final prediction is generated via **Logit Averaging (Soft Voting)** across all three models.

## Environment Setup
Recommended Environment: Python 3.9+ 

### Local Setup (Conda)
```bash
# Create environment
conda create -n VRDL python=3.9 -y
conda activate VRDL

# Install dependencies
pip install -r requirements.txt

├── data/
│   ├── train/
│   ├── val/
│   └── test/          
├── utils.py            # Utility functions
├── train_seed0.py      # Training script (Seed 0)
├── train_seed42.py     # Training script (Seed 42)
├── train_sgd.py        # Training script (SGD)
└── inference.py        # Ensemble inference script

python train_seed0.py
python train_seed42.py
python train_sgd.py
python inference.py

![Performance snapshot](snapshot.png)

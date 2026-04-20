# self-pruning-neural-network_Tredence_CaseStudy

# Self-Pruning Neural Network

## Overview
This project implements a neural network that learns to prune its own weights during training using learnable gate parameters and L1 sparsity regularization.

## Key Idea
Each weight is associated with a gate (0–1). During training:
- Important weights → gate ≈ 1
- Unimportant weights → gate → 0 (pruned)

## Loss Function
Total Loss = Classification Loss + λ × Sparsity Loss

## Results

| Lambda | Accuracy | Sparsity |
|--------|----------|----------|
| 0.5    | 50.48%   | 59.86%   |
| 1.0    | 49.86%   | 62.16%   |
| 2.0    | 49.79%   | 64.32%   |

## Observations
- Increasing λ increases sparsity
- Accuracy remains stable (~50%)
- Indicates redundancy in neural network weights

## How to Run

```bash
pip install -r requirements.txt
python train.py

---

# GitHub :

In terminal (VS Code):

```bash id="r8"
git init
git add .
git commit -m "Initial commit - self pruning NN"
git branch -M main
git remote add origin <your_repo_link>
git push -u origin main

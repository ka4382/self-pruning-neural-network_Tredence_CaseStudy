# Tredence_CaseStudy

# Self-Pruning Neural Network for Dynamic Model Compression

## Overview

This project implements a neural network that learns to prune its own weights during training. Traditional pruning methods remove weights after training is complete. In contrast, this approach integrates pruning into the training process itself, allowing the network to dynamically identify and eliminate less important connections.

The objective is to reduce model complexity while maintaining predictive performance, demonstrating that many parameters in neural networks are redundant.

---

## Problem Statement

Modern neural networks are often overparameterized, leading to inefficiencies in memory and computation. This project addresses the following:

- Can a neural network learn which of its own weights are unnecessary?
- Can we reduce the number of active parameters during training itself?
- How does pruning affect the trade-off between model accuracy and sparsity?

---

## Approach

A custom linear layer called `PrunableLinear` is implemented. Each weight in this layer is associated with a learnable gate parameter. The gate determines whether the weight is active or pruned.

The effective weight used during the forward pass is defined as:

Effective Weight = Weight × Sigmoid(Gate Score)

The sigmoid function ensures that gate values remain between 0 and 1.

- Gate value close to 1 → weight is active
- Gate value close to 0 → weight is effectively removed

This mechanism allows the model to learn which connections are important during training.

---

## Model Architecture

The model is a fully connected neural network trained on the CIFAR-10 dataset.

- Input: 32 × 32 × 3 images (flattened to 3072)
- Hidden Layer 1: 3072 → 512
- Hidden Layer 2: 512 → 256
- Output Layer: 256 → 10 classes

All layers use the custom `PrunableLinear` implementation instead of standard linear layers.

---

## Loss Function

The total loss used during training is:

Total Loss = Classification Loss + λ × Sparsity Loss

Where:

- Classification Loss: CrossEntropyLoss for image classification
- Sparsity Loss: Mean of gate values across all layers

The sparsity term encourages the model to minimize the number of active weights.

---

## Why L1-Based Sparsity Works

The sparsity loss is equivalent to applying L1 regularization on the gate values. L1 regularization encourages values to move toward zero. As the gate values approach zero, the corresponding weights are effectively pruned from the network.

This results in a sparse model where only the most important connections remain active.

---

## Training Details

- Optimizer: Adam
- Learning Rate: 0.0005
- Number of Epochs: 10
- Batch Size: 64
- Dataset: CIFAR-10

Multiple experiments were conducted with different values of λ to study the impact of sparsity regularization.

---

## Results

| Lambda | Accuracy | Sparsity |
|--------|----------|----------|
| 0.5    | 50.48%   | 59.86%   |
| 1.0    | 49.86%   | 62.16%   |
| 2.0    | 49.79%   | 64.32%   |

---

## Observations

- Increasing the value of λ leads to higher sparsity, indicating that more weights are being pruned.
- Despite removing over 60 percent of the weights, the model maintains nearly the same accuracy.
- This suggests that a significant portion of the network parameters are redundant.

---

## Key Insights

This experiment demonstrates that neural networks can be significantly compressed without a major drop in performance. The self-pruning mechanism allows the model to automatically adapt its structure during training.

The results highlight that:

- Neural networks are often overparameterized
- Many weights do not contribute meaningfully to performance
- Dynamic pruning can improve efficiency without sacrificing accuracy

---

## Gate Distribution Analysis

After training, the distribution of gate values shows two clear patterns:

- A large number of gates near zero, representing pruned weights
- A cluster of gates near one, representing important connections

This confirms that the model successfully learned to differentiate between useful and redundant weights.

---

## Project Structure

self-pruning-neural-network/

- model.py        : Contains PrunableLinear and network architecture
- train.py        : Training and evaluation logic
- utils.py        : Sparsity loss and evaluation utilities
- requirements.txt
- README.md

---

## How to Run

1. Clone the repository:

git clone https://github.com/your-username/self-pruning-neural-network.git  
cd self-pruning-neural-network

2. Install dependencies:

pip install -r requirements.txt

3. Run the training script:

python train.py

---

## Future Improvements

- Extend the approach to convolutional neural networks
- Explore structured pruning (removing entire neurons or filters)
- Apply the method to larger datasets and deeper architectures
- Combine pruning with quantization for further compression
- Deploy the compressed model on edge devices

---

## Conclusion

This project demonstrates a practical implementation of self-pruning neural networks using learnable gate parameters and sparsity regularization. The approach enables dynamic model compression during training and shows that significant reductions in model size can be achieved without compromising performance.

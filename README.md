# Blueberry-AI
Train a GPT-4-level LLM on the whole internet for $100 by 2029

## Overview
Blueberry-AI is an experimental large language model (LLM) project aiming to train a GPT-4-level model efficiently and cost-effectively. The goal is to optimize both speed and reasoning capabilities, exploring new architectures and optimization techniques.

## Features
- **Efficient Transformer Architecture**: Optimized for fast training.
- **Gradient Stabilization**: Prevents exploding gradients.
- **Configurable Context Length**: Supports long-form reasoning.
- **Multi-GPU & Distributed Training**: Scale training across multiple devices.
- **Automatic Mixed Precision (AMP)**: Memory-efficient training.
- **Built-in Evaluation Metrics**: Loss, perplexity, accuracy.
- **Modular Code Structure**: Easy experimentation and extension.
- **Optimizer Flexibility**: Supports AdamW and Muon.
- **Lightweight Inference**: Can run efficiently on Colab T4 GPU.

## Architecture
- **Decoder-Only Transformer**: Optimized for autoregressive language modeling.
- **Rotary Positional Embeddings (RoPE)**: Efficiently encodes positional information for long sequences.
- **Grouped-Query Attention (GQA)**: Reduces memory usage while maintaining attention quality.
- **Gated Linear Unit (GLU)**: Enhances feed-forward expressivity with SiLU activation.
- **RMSNorm Stabilization**: Applied pre- and post-attention/feed-forward for robust training.
- **Untied Embedding Weights**: Separate token embedding and output projection for flexibility.
- **Depth-Aware Weight Initialization**: Scales initialization variance by layer depth.
- **Zero-Initialization for Key Layers**: Stabilizes training for attention and MLP outputs.
- **Scaled Embedding Input**: Normalizes token embeddings by sqrt(d_model) for stability.

## Training
- **Muon Optimizer**: Orthogonalized momentum for efficient gradient updates.
- **Automatic Mixed Precision (AMP)**: Reduces memory usage and speeds up training.
- **Multi-GPU Support**: Enables distributed training for scalability.

# Usage

1. Clone the repository:
```bash
!git clone https://github.com/HosLak/Blueberry-LLM.git
```

2. Change into the project directory:
```
%cd Blueberry-LLM
```

3. Run training:
```
!python run_train.py
```

5. Run inference:
```
!python inference.py
```

# Contributing
Feel free to contribute new ideas, experiments, or optimizations. Check the issues and discussions tab for ongoing experiments.

# License
This project is licensed under the **MIT License** - see the LICENSE file for details.

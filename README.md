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
- **QK normalization**: Applied for better stability.
- **RMSNorm Stabilization**: Applied pre- and post-attention/feed-forward for robust training.
- **Untied Embedding Weights**: Separate token embedding and output projection for flexibility.
- **Depth-Aware Weight Initialization**: Scales initialization variance by layer depth. [Unlocking Transformer Learning: Weight Dispersion and a Novel Depth-Aware Initialization Strategy](https://medium.com/@hosseinlack123/unlocking-transformer-learning-weight-dispersion-and-a-novel-depth-aware-initialization-strategy-6e43dddb10a4)
- **Zero-Initialization for Key Layers**: Stabilizes training for attention and MLP outputs.
- **Scaled Embedding Input**: Normalizes token embeddings by sqrt(d_model) for stability.
- **Better LR Scheduler**: Improved learning rate scheduling.

## Training
- **Muon Optimizer**: Orthogonalized momentum for efficient gradient updates.
- **Automatic Mixed Precision (AMP)**: Reduces memory usage and speeds up training.
- **Multi-GPU Support**: Enables distributed training for scalability.
- **Dataset**: Uses the [Hosseinlack123/Blueberry-testdataset](https://huggingface.co/datasets/Hosseinlack123/Blueberry-testdataset) for training.
- **Tokenizer**: Employs a custom BPE tokenizer with a vocabulary size of 24,576 ([Hosseinlack123/Blueberry-testtokenizer](https://huggingface.co/Hosseinlack123/Blueberry-testtokenizer)).

Training Results

Using a free Google Colab T4 runtime and only 2 minutes 24 seconds of training, the following result was achieved:

Prompt: ```Once upon a time```

Model Output (trained 2m24s on T4):
```
Once upon a time, there was a little girl named Lucy. She had a very special wand that she loved to wear pretty things. One day, the garden became very gloomy in her garden and it was very cold, so she decided to go inside.

When they arrived home, Lucy showed her a bottle of perfume, some of water and leaves. The plant was so shiny and delicious! It smelled like an aeroplane, but couldn't see it anywhere. Lucy said it would be okay.

At first, Lucy looked for something else to eat for the melon, but it couldn't find the flower for dinner again. It was too sweet, and it started to feel its way to get more confused.

At the end, Lucy's mom asked
```
**Analysis**: The model already produces fluent grammar and some coherent story structure after a very short training run - impressive for 2m24s. However, the output clearly shows hallucination/nonsense (weird facts, repeated/confused phrases). Two main causes are identified:

- **Too little training time**: The short training duration limits the model's ability to learn complex patterns.
- **Too-small dataset**: The limited dataset size restricts the model's exposure to diverse language patterns.

# Model Configuration

The model is currently configured with the following hyperparameters for testing purposes:

- n_layers: 8
- hidden_dim: 256
- n_heads: 8 (head_dim: 32)
- n_kv_heads: 2
- vocab_size: 24,576
- seq_len: 256

**Note**: This is an experimental model designed to test and optimize the architecture's performance on a limited budget. For a deeper dive into the reasoning behind these hyperparameter choices and strategies for building efficient AI models, check out this article: [AI Unlocking Hyperparameter Secrets: Building Better AI Models on a Budget](https://medium.com/@hosseinlack123/aiunlocking-hyperparameter-secrets-building-better-ai-models-on-a-budget-60e25562b5e9)
Once the architecture is finalized, we plan to scale up to the following hyperparameters:

- n_layers: 24
- hidden_dim: 2048
- n_heads: 32 (head_dim: 64)
- n_kv_heads: 8
- vocab_size: Approximately equivalent to GPT-2's vocabulary size
- seq_len: Approximately 2048 (if increasing context length is feasible, otherwise likely to remain around 2048)


# Usage

1. Clone the repository:
```bash
!git clone https://github.com/HosLak/Blueberry-LLM.git
```

2. Change into the project directory:
```bash
%cd Blueberry-LLM
```

3. Run training:
```bash
!python run_train.py
```

4. Run inference:
```bash
!python inference.py
```

# Contributing
Feel free to contribute new ideas, experiments, or optimizations. Check the issues and discussions tab for ongoing experiments.

# License
This project is licensed under the **MIT License** - see the LICENSE file for details.

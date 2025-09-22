# Simple-PicoLM
Developing more efficient Transformer architectures for language tasks, aiming to drastically reduce model size while preserving high performance.

## Overview
Simple-PicoLM is an small language model (SLM) project designed to train GPT-like models efficiently and cost-effectively. Its goal is to balance speed and reasoning capabilities while exploring new architectures and optimization techniques. Unlike large and complex models, Simple-PicoLM emphasizes efficiency, simplicity, and low resource consumption, making it well-suited for both specific and general tasks, as well as custom tools or resource-constrained environments.

### Key Difference from the Main PicoLM:
- PicoLM is a more complete and commercial project with advanced features.
- simple-PicoLM is its open-source and non-commercial version, providing the base code to the community, but excluding some key architectural details (such as specific optimizations or advanced modules) to preserve intellectual property rights.

However, simple-PicoLM also performs well in general tasks like everyday chats or content generation—it doesn't fall short!

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
- **Open-Source**: Base code available on GitHub, with MIT license.

## Architecture
- **Decoder-Only Transformer**: Optimized for autoregressive language modeling.
- **Rotary Positional Embeddings (RoPE)**: Efficiently encodes positional information for long sequences.
- **Grouped-Query Attention (GQA)**: Reduces memory usage while maintaining attention quality.
- **Gated Linear Unit (GLU)**: Enhances feed-forward expressivity with SiLU activation.
- **PyramidNet MLP Structure**: Incorporates pyramidal MLPs to improve training efficiency by progressively increasing dimensionality in feed-forward layers, balancing computational cost and model capacity. See [PyramidNet: A Technique for Efficient Transformer Training with Pyramidal MLPs](https://medium.com/@hosseinlack123/pyramidnet-a-technique-for-efficient-transformer-training-with-pyramidal-mlps-a3caa85918ae) for details.
- **QK normalization**: Applied for better stability.
- **RMSNorm Stabilization**: Applied pre- and post-attention/feed-forward for robust training.
- **Untied Embedding Weights**: Separate token embedding and output projection for flexibility.
- **Depth-Aware Weight Initialization**: Scales initialization variance by layer depth, applied in reverse due to the PyramidNet MLP structure. See [Unlocking Transformer Learning: Weight Dispersion and a Novel Depth-Aware Initialization Strategy](https://medium.com/@hosseinlack123/unlocking-transformer-learning-weight-dispersion-and-a-novel-depth-aware-initialization-strategy-6e43dddb10a4) for details.
- **Zero-Initialization for Key Layers**: Stabilizes training for attention and MLP outputs.
- **Scaled Embedding Input**: Normalizes token embeddings by sqrt(d_model) for stability.
- **Better LR Scheduler**: Improved learning rate scheduling.
- **Gated Attention (SDPA Output G1)**: Applies a head-specific, elementwise sigmoid gate after Scaled Dot-Product Attention (SDPA) outputs, calculated based on Query (Q).

## Training
- **Muon Optimizer**: Orthogonalized momentum for efficient gradient updates.
- **Automatic Mixed Precision (AMP)**: Reduces memory usage and speeds up training.
- **Multi-GPU Support**: Enables distributed training for scalability.
- **Dataset**: Uses the [Hosseinlack123/PicoLM-dataset](https://huggingface.co/datasets/Hosseinlack123/PicoLM-dataset) for training.
- **Tokenizer**: Employs a custom BPE tokenizer with a vocabulary size of 24,576 ([Hosseinlack123/PicoLM-tokenizer](https://huggingface.co/Hosseinlack123/PicoLM-tokenizer)).

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

Once the architecture is finalized, I plan to scale up to the following hyperparameters:

- n_layers: 24
- hidden_dim: 2048
- n_heads: 32 (head_dim: 64)
- n_kv_heads: 8
- vocab_size: Approximately equivalent to GPT-2's vocabulary size
- seq_len: Approximately 2048 (if increasing context length is feasible, otherwise likely to remain around 2048)

# Future Roadmap

**We are actively working on improving the PicoLM-AI model.** Planned enhancements include:

- **Stepwise Think Fine-Tuning**: The model will be fine-tuned using the Stepwise Think format to enhance reasoning capabilities and achieve faster, high-quality responses. For more details on this approach, see [Stepwise Think: Revolutionizing Chain-of-Thought for Faster AI Responses Without Sacrificing Quality](https://medium.com/@hosseinlack123/stepwise-think-revolutionizing-chain-of-thought-for-faster-ai-responses-without-sacrificing-d81e140789b6).

**Additional plans will be added as the project evolves.**

# Usage

1. Clone the repository:
```bash
!git clone https://github.com/HosLak/PicoLM.git
```

2. Change into the project directory:
```bash
%cd PicoLM
```

3. Run training:
```bash
!python run_train.py
```

4. Run inference:
```bash
!python inference.py
```

#

Limitations

- This is a simplified version and lacks some advanced features of PicoLM (such as specific optimizations or support for very large models).
- For commercial use, refer to the main PicoLM project.
- The model may require additional tuning for highly complex tasks.

# License
This project is licensed under the **MIT License** - see the LICENSE file for details.

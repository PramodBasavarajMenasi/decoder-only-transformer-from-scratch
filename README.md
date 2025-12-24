

# 🧠 Decoder-Only Transformer from Scratch

A **decoder-only Transformer language model built from scratch**, designed to study **modern LLM architecture choices**, **compute-aware design**, and **practical evaluation beyond training loss** under limited GPU resources.

This project focuses on *understanding how real-world decoder-only language models work*, rather than maximizing scale.

---

## 🔍 Project Overview

This repository implements and trains a **decoder-only autoregressive Transformer** with modern components commonly used in large language models, including:

- 🧩 Byte Pair Encoding (BPE) tokenization  
- 🔒 Causal self-attention  
- 🔄 Rotary Positional Encoding (RoPE)  
- 🧠 Grouped-Query Attention (GQA)  
- ⚡ SwiGLU feed-forward layers  

The model was intentionally trained at **small scale** to validate architectural decisions, analyze trade-offs, and observe generation behavior under realistic compute constraints.

---

## 🧠 Architecture

### Model Type
- **Decoder-only Transformer**
- Autoregressive, left-to-right text generation
- Causal masking to prevent future token leakage

### Core Components

| Component | Description | Why Used |
|--------|------------|---------|
| BPE Tokenization | Subword tokenization | Handles open vocabulary efficiently while keeping sequence length manageable |
| Causal Self-Attention | Left-to-right attention | Required for autoregressive generation |
| Grouped-Query Attention (GQA) | Shared key–value heads | Reduces memory and compute vs standard multi-head attention |
| Rotary Positional Encoding (RoPE) | Relative position encoding | Improves generalization to longer contexts |
| SwiGLU Feed-Forward Layers | Gated activation | Better expressiveness and gradient flow than standard FFN/ReLU |

---

### 📐 Architecture Diagram


<img width="1024" height="512" alt="image" src="https://github.com/user-attachments/assets/2552ea9b-3398-42ce-a6a5-e6129accf11b" />



> Decoder-only Transformer with RoPE-based Grouped-Query Attention (GQA) and MoE-SwiGLU feed-forward layers.  
> Configuration: 5 decoder blocks, hidden size 384, vocabulary size 10,000.

---

### 🧩 Model Summary

```text
TextGenerationModel(
  (rope): RotaryPositionalEncoding()
  (embedding): Embedding(10000, 384)
  (decoder): ModuleList(
    (0-4): 5 x Decoder(
      (self_atten): GQA(
        (q_proj): Linear(384 → 384)
        (k_proj): Linear(384 → 128)
        (v_proj): Linear(384 → 128)
        (out_proj): Linear(384 → 384)
      )
      (mlp): MoELayer(
        (experts): 2 × SwiGLU(
          384 → 1536 → 384
        )
        (router): Linear(384 → 2)
      )
      (norm1): RMSNorm(384)
      (norm2): RMSNorm(384)
    )
  )
  (final_norm): RMSNorm(384)
  (output): Linear(384 → 10000)
)
```

## ⚙️ Model Configuration (Example)

```python
model_config = {
    "num_layers": 5,
    "num_heads": 6,
    "num_kv_heads": 2,
    "hidden_dim": 384,
    "max_seq_len": 256,
    "vocab_size": tokenizer_vocab_size,
    "dropout": 0.1
}
````

> Multiple configurations were tested to study depth–width trade-offs and their impact on training stability, efficiency, and text generation quality.

---

## ☁️ Training Environment

* Platform: **Google Colab**
* GPU: **NVIDIA T4**
* Framework: **PyTorch**
* Precision: **Mixed Precision (FP16)**

---

## 🚀 Training

### Training Philosophy

* Architecture validation
* Stable optimization
* Qualitative generation analysis
* Compute-aware experimentation

### Key Training Choices

* Mixed precision
* Gradient clipping
* Step-limited or single-epoch training
* Compute-aware batch size and sequence length

---

## 📊 Evaluation Approach

* Training loss for optimization health
* Qualitative generation analysis
* Token and subword stability
* Sentence structure and coherence
* Long-range dependency handling

---

## 📌 Key Learnings

* Training loss does not fully reflect generation quality
* Model capacity strongly impacts syntactic coherence
* GQA, RoPE, and SwiGLU improve efficiency and stability
* Compute-aware design enables realistic prototyping

---

## 🧪 Why Small-Scale Training?

* Validate architectural choices
* Explore efficiency trade-offs
* Understand decoder-only behavior
* Enable rapid iteration

---

## 📚 References

* *Attention Is All You Need*
* Rotary Positional Embeddings (RoPE)
* Grouped-Query Attention (GQA)
* SwiGLU

---


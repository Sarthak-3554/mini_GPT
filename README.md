# Mini-GPT

A small GPT-style language model built **from scratch in PyTorch**, including a custom BPE tokenizer, pretraining pipeline, autoregressive generation, and KV-cache accelerated inference.

The project focuses on understanding the internals of a modern decoder-only Transformer rather than relying on an existing pretrained LLM.

---

## Highlights

- GPT-style decoder-only Transformer implemented from scratch
- ~6.57M parameter language model
- Custom Byte-Level BPE tokenizer
- 8,000-token vocabulary
- Trained from scratch on the Shakespeare corpus
- Grouped Query Attention (GQA)
- Rotary Positional Embeddings (RoPE)
- RMSNorm
- SwiGLU feed-forward network
- Weight tying between token embeddings and LM head
- Temperature and Top-K sampling
- Instruction fine-tuning pipeline
- Experimental LoRA implementation
- KV-cache based autoregressive inference
- Numerical correctness testing for KV cache
- Inference benchmarking on Apple Silicon MPS

---

## Model Architecture

The model is a decoder-only Transformer with the following components:

```text
Input Tokens
     │
     ▼
Token Embedding
     │
     ▼
┌─────────────────────────────┐
│     Transformer Block       │
│                             │
│   RMSNorm                   │
│      │                      │
│      ▼                      │
│   GQA Attention             │
│      │                      │
│      ▼                      │
│   Residual Connection       │
│      │                      │
│   RMSNorm                   │
│      │                      │
│      ▼                      │
│   SwiGLU                    │
│      │                      │
│      ▼                      │
│   Residual Connection       │
└─────────────────────────────┘
             │
          × N Layers
             │
             ▼
        Final RMSNorm
             │
             ▼
        LM Head
             │
             ▼
       Next Token
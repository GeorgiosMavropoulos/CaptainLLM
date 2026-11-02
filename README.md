# Building an LLM From Scratch → GPT-2 Classifier & Instruction Model

> **A hands-on implementation of a GPT-2-style decoder-only Transformer in PyTorch, adapted into a text classifier and an instruction-following model through fine-tuning.**

## 📌 Project Overview

This project explores how Large Language Models work by implementing a GPT-2-style decoder-only Transformer from the ground up using **Python and PyTorch**.

It starts with the fundamental components of a language model (text processing, tokenization, embeddings, self-attention, Transformer blocks, training, and text generation) and then adapts the pretrained GPT-2 model for two downstream tasks:

1. **GPT-2 Classifier**: the model is fine-tuned for text classification.
2. **GPT-2 Instruction Model**: the model is fine-tuned to follow natural-language instructions.

```text
┌─────────────────────────────────────┐
│   Stage 1 — LLM From Scratch        │
│                                     │
│ Text → Tokens → Transformer → GPT-2 │
│                                     │
│   Pretraining & Text Generation     │
│   Loading pretrained GPT-2 weights  │
└──────────────────┬──────────────────┘
                   │
         ┌─────────┴──────────┐
         ▼                    ▼
┌─────────────────┐  ┌─────────────────────┐
│  Stage 2        │  │  Stage 3            │
│  GPT-2          │  │  GPT-2              │
│  Classifier     │  │  Instruction Model  │
│                 │  │                     │
│ Classification  │  │ Instruction         │
│ head + fine-    │  │ dataset + SFT       │
│ tuning          │  │                     │
└─────────────────┘  └─────────────────────┘
```

The goal is **not** to reproduce a production-scale model. Instead, the goal is to understand the underlying mechanisms by implementing them, experimenting with them, and measuring their behavior.

---

# 🎯 Objectives

* Understand how modern autoregressive language models work.
* Implement the core components of a GPT-2-style Transformer.
* Understand tokenization and vocabulary construction.
* Implement self-attention and multi-head attention.
* Understand causal language modeling.
* Build a complete training pipeline in PyTorch.
* Load pretrained GPT-2 weights into the custom implementation.
* Fine-tune GPT-2 as a **classifier** by replacing the output head.
* Fine-tune GPT-2 as an **instruction-following model**.
* Evaluate both models with appropriate metrics.
* Investigate how architectural and training decisions affect performance.

---

# 📚 Learning Foundation

The implementation is developed while studying:

**Sebastian Raschka — *Build a Large Language Model (From Scratch)***

The book provides the conceptual and implementation foundation for the GPT-2 model, the classification fine-tuning, and the instruction fine-tuning. The purpose of referencing the book is to make the learning path transparent and distinguish the foundational implementation from later project-specific work and experiments.

---

# 🗺️ Development Roadmap

## Phase 1 — Text Processing

* [x] Load initial text corpus
* [x] Basic tokenization
* [x] Vocabulary construction
* [x] Token → ID mapping
* [x] ID → Token mapping
* [x] Dataset creation
* [x] Context-window generation

## Phase 2 — Transformer Architecture

* [x] Token embeddings
* [x] Positional embeddings
* [x] Self-attention
* [x] Causal masking
* [x] Multi-head attention
* [x] Feed-forward network
* [x] Layer normalization
* [x] Residual connections
* [x] Transformer block
* [x] GPT-2-style model

## Phase 3 — Pretraining & Generation

* [x] Training loop
* [x] Cross-entropy loss
* [x] Optimizer
* [x] Validation loop
* [x] Loss visualization
* [x] Text generation and sampling
* [x] Loading pretrained GPT-2 weights

## Phase 4 — GPT-2 Classifier

* [x] Prepare a labeled classification dataset
* [x] Replace the language-modeling head with a classification head
* [x] Freeze / unfreeze selected layers
* [x] Fine-tune the classifier
* [x] Evaluate accuracy on train / validation / test splits
* [ ] Report detailed results (precision, recall, F1)

## Phase 5 — GPT-2 Instruction Model

* [x] Prepare an instruction dataset
* [x] Format instruction–response pairs with a prompt template
* [x] Build a batching / padding / masking pipeline
* [x] Instruction fine-tuning
* [x] Generate responses to unseen instructions
* [ ] Evaluate response quality (e.g. LLM-as-a-judge or benchmark)

---

# 🧠 Model Architecture

The base model follows the GPT-2 decoder-only Transformer architecture.

```text
Input Tokens
     │
     ▼
Token Embeddings
     │
     +
Positional Embeddings
     │
     ▼
┌──────────────────────┐
│ Transformer Block    │
│                      │
│ LayerNorm            │
│     ↓                │
│ Causal Self-Attention│
│     ↓                │
│ Residual Connection  │
│     ↓                │
│ LayerNorm            │
│     ↓                │
│ Feed Forward Network │
│     ↓                │
│ Residual Connection  │
└──────────┬───────────┘
           │
           ▼
       Repeat N times
           │
           ▼
       Final LayerNorm
           │
     ┌─────┴───────────────┐
     ▼                     ▼
 LM Head                Classification Head
 (vocab-size logits)    (num-classes logits)
     │                     │
     ▼                     ▼
 Next Token /          Predicted Class
 Instruction Response
```

The implementation intentionally exposes these components rather than hiding them behind a high-level model library.

---

# 🏷️ GPT-2 Classifier

The classifier adapts the pretrained GPT-2 model for text classification.

**Approach**

* The original output layer (projecting to the vocabulary) is replaced with a small linear layer projecting to the number of classes.
* The classification is made from the **last token's** output, which, thanks to causal attention, has attended to the entire input sequence.
* Only selected parts of the model (the new head, the final Transformer block, and the final LayerNorm) are fine-tuned, while the rest is frozen.
* Training uses cross-entropy loss over the class labels.

```text
Text → Tokenizer → GPT-2 → Last-token hidden state → Linear → Class logits
```

**Results**

| Split      | Loss | Accuracy |
| ---------- | ---: | -------: |
| Training   |  TBD |      TBD |
| Validation |  TBD |      TBD |
| Test       |  TBD |      TBD |

---

# 💬 GPT-2 Instruction Model

The instruction model fine-tunes GPT-2 to follow natural-language instructions through supervised fine-tuning (SFT).

**Approach**

* Training examples are instruction–(input)–response triples formatted with a consistent prompt template.
* A custom collate function handles padding and target masking.
* The model is trained with next-token prediction on the formatted examples.
* After fine-tuning, the model generates responses to unseen instructions.

```text
### Instruction:
<task description>

### Input:
<optional context>

### Response:
<model output>
```

**Example**

```text
Instruction: Rewrite the sentence in the passive voice.
Input:       The chef cooked the meal.

Response:    TBD
```

**Results**

| Metric                | Result |
| --------------------- | -----: |
| Training loss         |    TBD |
| Validation loss       |    TBD |
| Response quality      |    TBD |

---

# 🔬 Experiments

The project places emphasis on experimentation rather than just obtaining working models.

### Model Size

Compare GPT-2 variants (e.g. 124M and larger) in terms of layers, attention heads, embedding dimensions, and parameters.

### Classifier Experiments

* Fine-tuning only the head vs. head + last block vs. the full model
* Effect of the pooled token (last token vs. others)
* Effect of padding and maximum sequence length

### Instruction-Tuning Experiments

* Effect of dataset size and quality
* Effect of prompt template
* Effect of learning rate, batch size, weight decay, warm-up, and number of epochs
* Sampling parameters (temperature, top-k) on response quality

Every significant experiment should be reproducible and documented.

---

# 🛠️ Project Structure

```text
.
├── README.md
│
├── src/
│   ├── tokenizer/
│   ├── model/
│   │   ├── attention.py
│   │   ├── transformer.py
│   │   └── gpt.py
│   ├── data/
│   ├── training/
│   ├── classifier/
│   ├── instruction/
│   └── generation/
│
├── experiments/
│   ├── baseline/
│   ├── classifier/
│   └── instruction_tuning/
│
├── notebooks/
├── configs/
├── tests/
├── checkpoints/
├── results/
└── report/
```

The exact structure may evolve as the project grows.

---

# 📊 Evaluation

## Language Modeling

* training loss
* validation loss
* perplexity

```text
PPL = exp(cross_entropy_loss)
```

## Classifier

* accuracy on train / validation / test sets
* precision, recall, and F1 where appropriate
* confusion matrix

## Instruction Model

* validation loss on held-out instruction data
* qualitative inspection of generated responses
* automated scoring of responses (e.g. using another LLM as a judge) or benchmark evaluation

Lower loss does not automatically imply better responses, so quantitative metrics are complemented by inspection of actual outputs.

---

# 📈 Results Summary

| Model                  | Parameters | Task                     | Key Metric       | Result |
| ---------------------- | ---------: | ------------------------ | ---------------- | -----: |
| GPT-2 (from scratch)   |        TBD | Language modeling        | Validation loss  |    TBD |
| GPT-2 Classifier       |        TBD | Text classification      | Test accuracy    |    TBD |
| GPT-2 Instruction Model|        TBD | Instruction following    | Response quality |    TBD |

Results are accompanied by details about the experimental setup so that comparisons are meaningful.

---

# 🔧 Technologies

* Python
* PyTorch
* NumPy
* tiktoken (GPT-2 BPE tokenizer)
* Git / GitHub

---

# ⚙️ Reproducibility

Each experiment should record:

```text
Model configuration
Dataset/version
Random seed
Training parameters
Hardware
Software environment
Training duration
Evaluation procedure
Results
```

Configuration files are used where appropriate to avoid hard-coding experimental settings.

---

# 📝 Technical Report

A technical report accompanies the implementation and documents:

1. Background and motivation
2. Transformer / GPT-2 architecture
3. Tokenization
4. Dataset preparation
5. Model implementation
6. Pretraining and loading pretrained weights
7. Classification fine-tuning
8. Instruction fine-tuning
9. Evaluation methodology
10. Experimental results
11. Limitations
12. Future work

Unsuccessful experiments and implementation challenges are also discussed where they provide useful insight.

---

# ⚠️ Limitations

This project is an educational and experimental implementation.

* relatively small model size (GPT-2 scale)
* limited training compute
* small fine-tuning datasets
* limited context length
* the instruction model can produce incorrect, inconsistent, or unsafe responses
* limited evaluation coverage for the instruction model

These limitations should be considered when interpreting the results.

---

# 🚀 Future Work

* Adapt the model to **Python code generation** (code-specific pretraining and fine-tuning, with functional evaluation such as pass@k)
* Larger models and higher-quality datasets
* Preference tuning (e.g. DPO)
* LoRA / parameter-efficient fine-tuning
* Quantization and optimized inference
* Better evaluation of instruction following
* Retrieval-augmented generation
* Modern Transformer architectural improvements

---

# 📖 References

### Primary Learning Resource

Sebastian Raschka,
***Build a Large Language Model (From Scratch)***

Official companion material:

https://www.oreilly.com/library/view/build-a-large/9781633437166/

### Additional References

Additional papers, documentation, datasets, and technical resources used during development will be documented here.

---

# 👤 About This Project

This repository documents the process of learning how Large Language Models work by **building one rather than treating the model as a black box**.

The project progresses from simple components to a complete GPT-2 model, and then to two specialized systems: a **text classifier** and an **instruction-following model**.

The emphasis is on:

**understanding → implementation → experimentation → evaluation**

---

## ⭐ Project Status

✅ GPT-2 architecture, pretraining pipeline, **classifier**, and **instruction model** implemented.

🚧 Ongoing: detailed evaluation, experiments, technical report, and the extension to code generation.

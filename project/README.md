# Project — Multimodal Tool Retrieval with ToolBench

**Course:** Multimodal AI (MAS.S60 / 6.S985) · Spring 2026 · MIT
**Team:** Michael Serrano, Arthur De Los Santos, Dylan Mazard

## Overview

This project proposes a multimodal approach to agentic tool-use retrieval, treating natural language (user queries) and structured data (API schemas, code snippets) as distinct modalities. Using the ToolBench dataset, we build a retrieval system that accurately predicts the correct API calls for a given instruction by bridging the semantic gap between human intent and rigid, deterministic tool environments.

## Research Ideas

### Idea 1: Hierarchical Contrastive Alignment
ToolBench APIs are organized hierarchically (API → Tool → Category). We propose a **Hierarchical Contrastive Loss** that simultaneously aligns user intent with fine-grained API documentation *and* coarse category labels:

$$\mathcal{L} = \lambda_1 \cdot L_{\text{contrastive}}(I, D) + \lambda_2 \cdot L_{\text{contrastive}}(I, C)$$

**Hypothesis:** Category-level alignment acts as a regularizer that improves Recall@k for tail APIs with sparse documentation.

### Idea 2: Code-Enhanced Tri-Modal Alignment
We propose a **Tri-Encoder** architecture that fuses three modalities in a shared latent space:
- **Encoder A (BERT):** User instruction $I$
- **Encoder B (BERT):** API documentation $D$
- **Encoder C (CodeBERT):** Functional code snippet $K$

**Hypothesis:** Including executable code disambiguates APIs that are semantically similar in text but functionally different in practice.

### Idea 3: Hard-Negative Mining via DFSDT Solution Paths
Instead of random negatives, we mine **hard negatives** from ToolBench's Depth-First Search Decision Tree (DFSDT) traces — APIs that were considered but rejected during successful trajectories.

**Hypothesis:** Training on failure-path negatives teaches the model to distinguish functionally similar tools (e.g., `get_current_weather` vs. `get_weather_forecast`).

## Metrics

- **Recall@k** (k = 1, 5, 10) — primary retrieval metric
- **Mean Reciprocal Rank (MRR)**
- **API Selection F1**

## Repository Structure

```
project/
├── README.md           ← this file
├── data/               ← data loading and preprocessing scripts
├── models/             ← encoder architectures
├── training/           ← training loops and loss functions
├── evaluation/         ← metric computation
└── notebooks/          ← experiment notebooks
```

## Running the Experiments

All experiments run on Google Colab Pro. Follow notebooks in order:

1. **[01_index_apis.ipynb](notebooks/01_index_apis.ipynb)** — Pre-compute OpenAI embeddings for all API docs. Run once; results are cached.
2. **[02_baseline_eval.ipynb](notebooks/02_baseline_eval.ipynb)** — Baseline retrieval evaluation (Recall@k, MRR).
3. **[03_hard_negative_eval.ipynb](notebooks/03_hard_negative_eval.ipynb)** — Hard-negative ablation across random, category-sibling, and DFSDT failure-path conditions.

### Requirements
- Google Colab Pro (or any GPU/CPU runtime)
- OpenAI API key (set as Colab secret `OPENAI_API_KEY`)
- ToolBench dataset in a `toolbench/` directory

### Results
See `results_baseline.json` and `results_hard_negatives.json` for raw numbers.

## Reports

- [Proposal](../latex/Project_Proposal__Modeling_For_MultiModal_AI/main.tex)
- [Midterm Report](../latex/Project_Midterm_Report/)

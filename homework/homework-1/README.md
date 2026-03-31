# Homework 1 — Dataset: Multimodal Data Preprocessing with ToolBench

**Course:** Multimodal AI (MAS.S60 / 6.S985) · Spring 2026 · MIT

## Overview

This assignment focuses on loading, parsing, and exploring the [ToolBench](https://github.com/OpenBMB/ToolBench) dataset as a multimodal data source. ToolBench contains real-world agent trajectories where an LLM uses over 16,000 APIs across 50 categories to answer user instructions.

## Modalities Explored

| Modality | Description |
|---|---|
| **Natural Language** | User queries and system prompts |
| **Structured API Calls** | Function names and parameter key-value pairs |
| **API Responses** | JSON-structured outputs and error signals |

## Key Work

- **Data loading & parsing** — loaded `toolllama_G123_dfs_train.json` and `toolllama_G123_dfs_eval.json` alongside 10,657 tool definitions across 50 categories
- **Modality extraction pipeline** — built `extract_modalities()` to decompose each conversation into user queries, reasoning steps, function calls, and responses
- **Distribution analysis** — histograms and box plots showing per-modality statistics across 5,000 training examples
- **t-SNE visualizations** — TF-IDF + t-SNE projections showing that each modality (queries, calls, responses) occupies a distinct region of input space
- **Metric implementations** — implemented Pass Rate, API Selection F1, and Recall@k from scratch as evaluation baselines

## Notebook

[`mys1_mmai_HW1.ipynb`](./mys1_mmai_HW1.ipynb)

# Homework 2 — Fusion: Multimodal Alignment and Fusion Techniques

**Course:** Multimodal AI (MAS.S60 / 6.S985) · Spring 2026 · MIT

## Overview

This assignment explores multimodal alignment and fusion using the AV-MNIST dataset (audio + vision), building on the [MultiBench](https://github.com/pliang279/MultiBench) framework.

## Topics Covered

- **Tensors & Einsum** — foundational tensor operations used throughout multimodal pipelines (dot products, transposes, element-wise and column-wise summations)
- **Unimodal Models** — separate audio and image models trained independently as baselines before fusion
- **Multimodal Fusion** — fusing audio and visual streams using techniques from MultiBench (concatenation, multiplicative interactions)

## Dataset

**AV-MNIST** — a multimodal version of MNIST pairing each digit image with a spoken audio recording of the digit.

| Split | Purpose |
|---|---|
| Train | Model training |
| Validation | Hyperparameter tuning |
| Test | Final evaluation |

## Notebook

[`mys1_mmai_HW2.ipynb`](./mys1_mmai_HW2.ipynb)

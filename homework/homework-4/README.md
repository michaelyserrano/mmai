# Homework 4 — GRPO: Reinforcement Learning for Vision-Language Models

**Course:** Multimodal AI (MAS.S60 / 6.S985) · Spring 2026 · MIT

## Overview

This assignment explores **Reinforcement Learning for Vision-Language Models**, implementing and training a VLM using **Group Relative Policy Optimization (GRPO)** — a recent RL algorithm for aligning language models with reward signals, introduced in DeepSeekMath.

## Topics Covered

- Understanding GRPO vs. PPO: group-based advantage estimation without a learned critic
- Reward design: rule-based vs. learned reward models, and reward hacking risks
- Comparing SFT (Homework 3) with RL-based fine-tuning (GRPO)
- Implementing group-relative advantage computation from scratch
- Defining reward functions for structured VLM outputs
- Building a training dataset formatted for the GRPO RL loop
- Training a VLM with GRPO using the TRL library
- Post-training evaluation on held-out images

## Structure

| Part | Description | Points |
|---|---|---|
| Part 1 | Reading & Reflection (GRPO theory, reward design, SFT vs. GRPO) | 20 |
| Problem 1 | GPU verification and library installation | — |
| Problem 2 | Prepare dataset (reuse from HW3, train/test split) | 10 |
| Problem 3 | Understanding GRPO (group advantage, clipping) | 15 |
| Problem 4 | Implement GRPO advantage computation | 25 |
| Problem 5 | Define reward functions | 15 |
| Problem 6 | Build training dataset for GRPO | 10 |
| Problem 7 | Train with GRPO | 20 |
| Problem 8 | Post-training evaluation and reflection | 20 |

## Notebook

[`mys1_mmai_HW4.ipynb`](./mys1_mmai_HW4.ipynb)

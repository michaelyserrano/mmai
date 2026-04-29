# Homework 5 — Agents: Building, Evaluating, and Improving a Goal-Directed Agent

**Course:** Multimodal AI (MAS.S60 / 6.S985) · Spring 2026 · MIT

## Overview

This assignment covers the design and implementation of an AI agent that operates in an environment, takes actions over multiple steps, and attempts to accomplish user-defined goals. The agent is built over a mocked sports API environment and evaluated on a 12-task benchmark spanning normal, edge, and ambiguous cases.

## Topics Covered

- Formalizing agent tasks as sequential decision-making problems
- Implementing an agent loop with observations, actions, state updates, and termination conditions
- Designing and exposing custom tools through a structured interface (smolagents)
- Evaluating agent behavior with offline and online benchmarks
- Analyzing failures by type: routing, reasoning, policy, and tool execution
- Multimodal document QA with text-only vs. vision-enhanced agents
- Agent observability with Langfuse (tracing, latency, token cost)
- Safety and policy evaluation with before/after mitigation comparison

## Notebook

[`mys1_mmai_HW5.ipynb`](./mys1_mmai_HW5.ipynb)

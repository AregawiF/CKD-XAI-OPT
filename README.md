# Benchmarking Nature-Inspired Metaheuristics for Minimal Sufficient XAI Explanations

[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the source code for a comparative study evaluating **Nature-Inspired Metaheuristics** against State-of-the-Art (SOTA) baselines in the domain of **Explainable AI (XAI)**. 

The primary goal is to identify **Minimal Sufficient Feature Subsets**—the smallest set of features that consistently preserves a model's prediction and the stability of its explanation profile.

---

## 📋 Table of Contents
- [Project Overview](#-project-overview)
- [Dataset & Model](#-dataset--model)
- [Fitness Function](#-fitness-function)
- [Implemented Algorithms](#-implemented-algorithms)
- [Benchmarking Results](#-benchmarking-results)
- [Statistical Significance](#-statistical-significance)
- [System Specifications](#-system-specifications)
- [Usage](#-usage)
- [Authors](#-authors)

---

## 🚀 Project Overview

Standard XAI methods like **SHAP** and **LIME** often produce "dense" explanations, assigning importance to every input feature. This complexity hinders human interpretability in clinical settings. This project implements an optimization suite to prune redundant features while maintaining high explanation fidelity.

**Key Contributions:**
*   **Pearson Correlation Fitness:** An upgraded fitness function prioritizing feature ranking stability over raw Euclidean distance.
*   **Multi-Paradigm Benchmarking:** Evaluation of Evolutionary (GA), Swarm (PSO), and Physics-based (SA) heuristics.
*   **SOTA Comparison:** Direct benchmarking against SHAP and LIME baselines.
*   **Statistical Rigor:** Analysis based on 30 independent runs per algorithm.

---

## 📊 Dataset & Model

-   **Dataset:** [UCI Chronic Kidney Disease (CKD)](https://archive.ics.uci.edu/dataset/336/chronic+kidney+disease)
-   **Instances:** 400 | **Features:** 25 (post-encoding)
-   **Black-Box Model:** Random Forest Classifier (100 trees, max depth = 10)
-   **Model Performance:** 
    - Test Accuracy: **98.33%**
    - Test F1-Score: **0.9867**

---

## 🧬 Fitness Function

We minimize a multi-objective fitness function $f(m)$:
$$f(m) = \alpha \cdot |\hat{p}_{\text{full}} - \hat{p}_{\text{mask}}| + \beta \cdot (1 - \rho(S_{full}, S_{mask})) + \gamma \cdot (\text{Sparsity})$$

- **Prediction Fidelity:** Penalizes changes in the model's positive-class probability between the full and masked feature sets.
- **Explanation Fidelity ($\rho$):** Uses **Pearson Correlation** to ensure the "narrative" of the explanation remains stable.
- **Sparsity:** Penalizes the number of features selected to ensure a minimal subset.

---

## 🤖 Implemented Algorithms

| Algorithm | Category | Hyperparameters |
| :--- | :--- | :--- |
| **Genetic Algorithm (GA)** | Evolutionary | Pop: 50, Crossover: 0.8, Mutation: 0.1 |
| **Particle Swarm (PSO)** | Swarm Intelligence | Particles: 30, $w$: 0.7, $c_1, c_2$: 1.5 |
| **Simulated Annealing (SA)** | Physics-based | $T_0$: 100, Cooling: 0.95, Iter/Temp: 10 |

---

## 🏆 Benchmarking Results

We benchmarked our best-performing heuristic (**PSO**) against the two most common SOTA baselines:

| Approach | Fitness (Lower is Better) | Improvement | Features Used |
| :--- | :--- | :--- | :--- |
| **Top-K SHAP Baseline** | 0.068397 | Baseline | 17 |
| **LIME Baseline** | 0.195365 | -185.6% | 15 |
| **Metaheuristic (PSO)** | **0.040559** | **+40.70%** | **6** |

**Conclusion:** The PSO optimization found a synergistic subset that is **40.70% more faithful** to the original model logic than SHAP ranking and **79.24% better** than LIME, while using 60% fewer features.

---

## 📈 Statistical Significance

Results from **30 independent runs** were validated using **Wilcoxon Rank-Sum tests** ($p < 0.05$):

*   **PSO vs SOTA (SHAP):** $p < 0.001$ (Statistically Superior)
*   **PSO vs LIME:** $p < 0.001$ (Statistically Superior)
-   **GA vs PSO:** $p < 0.001$ (PSO statistically superior)
-   **GA vs SA:** $p < 0.001$ (GA statistically superior)
-   **PSO vs SA:** $p < 0.001$ (PSO statistically superior)

**Computational Efficiency:**
1.  **Simulated Annealing:** Fastest (~25.66s avg)
2.  **PSO:** ~1.75x slower than SA (~44.97s avg)
3.  **GA:** ~2.82x slower than SA (~72.28s avg)

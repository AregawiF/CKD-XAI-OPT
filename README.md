# Explainable AI Feature Selection using Nature-Inspired Heuristics

This repository contains the implementation of an AI academic project for optimizing feature subsets in **Explainable AI (XAI)** using **nature-inspired heuristics**. The goal is to minimize the number of features while preserving model predictions and SHAP-based explanations.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Model](#model)
- [SHAP Explanations](#shap-explanations)
- [Optimization Problem](#optimization-problem)
- [Implemented Algorithms](#implemented-algorithms)
- [Results](#results)
- [Usage](#usage)
- [Folder Structure](#folder-structure)
- [Next Steps](#next-steps)

---

## Project Overview

This project explores whether **nature-inspired heuristics (NIAs)** can find minimal feature subsets that preserve the behavior of a machine learning model. The selected topic is **Explainable AI (XAI)**, with the **UCI Heart Disease dataset** as the testbed.

The project involves:

1. Training a black-box classifier (Random Forest).
2. Computing **local SHAP explanations** for a target instance.
3. Defining a **fitness function** that balances:
   - Prediction fidelity
   - SHAP fidelity
   - Sparsity
4. Using optimization algorithms to select feature subsets:
   - **Genetic Algorithm (GA)**
   - **Particle Swarm Optimization (PSO)**
   - **Simulated Annealing (SA)**

---

## Dataset

- **Name:** UCI Heart Disease
- **Instances:** 303
- **Features:** 13 numeric features
- **Target:** Binary (presence/absence of heart disease)

The dataset is loaded using the `ucimlrepo` Python library. Missing values are handled by replacing with column means.

---

## Model

- **Algorithm:** Random Forest Classifier
- **Parameters:** 100 trees, max depth = 5
- **Performance:**
  - Test Accuracy: 0.8352
  - Test F1-Score: 0.8148

This model serves as the black-box for XAI analysis.

---

## SHAP Explanations

- SHAP is used to compute **local explanations** for a selected instance.
- The **baseline SHAP values** are stored and reused for all fitness evaluations.
- Unselected features are masked by replacing their values with **training set means**.

---

## Optimization Problem

- **Search Space:** Binary vectors of length 13
- **Constraint:** At least one feature must be selected
- **Objective:** Minimize a fitness function combining:
  - Prediction fidelity
  - SHAP fidelity
  - Sparsity penalty

---

## Implemented Algorithms

1. **Genetic Algorithm (GA)**
   - Population-based evolutionary search
   - Crossover rate: 0.8
   - Mutation rate: 0.1

2. **Binary Particle Swarm Optimization (PSO)**
   - Swarm-based search in binary space
   - Cognitive & social coefficients: 1.5
   - Inertia weight: 0.7

3. **Simulated Annealing (SA)**
   - Physics-inspired search
   - Initial temperature: 100.0
   - Final temperature: 0.01
   - Cooling rate: 0.95

- All algorithms enforce the constraint of **at least one selected feature**.
- Each algorithm can run multiple independent runs for **statistical analysis**.

---

## Results (Example for one instance)

- **Best algorithm:** GA
- **Selected features:** `['age', 'cp', 'trestbps', 'chol', 'restecg', 'thalach', 'exang', 'oldpeak', 'thal']`
- **Prediction fidelity:** Full vs Masked prediction identical
- **SHAP fidelity:** Normalized L2 error = 0.0
- **Number of selected features:** 9 / 13

> Note: All three algorithms converged to the same solution for the selected instance.

- Multiple runs (`N_RUNS = 20`) are implemented to generate statistical summaries and boxplots of fitness and number of selected features.

---

## Usage

1. Install dependencies:

```bash
pip install -r requirements.txt

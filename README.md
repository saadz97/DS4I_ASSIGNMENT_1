# Predicting Forecasted Avalanche Hazard with Neural Networks

[![License](https://img.shields.io/badge/License-UCT_Academic-blue.svg)](https://github.com/saadz97/DS4I_ASSIGNMENT_1/blob/main/LICENSE)
[![Last Commit](https://img.shields.io/github/last-commit/saadz97/DS4I_ASSIGNMENT_1?color=blue)](https://github.com/saadz97/DS4I_ASSIGNMENT_1/commits/main)
[![Issues](https://img.shields.io/github/issues/saadz97/DS4I_ASSIGNMENT_1?color=blue)](https://github.com/saadz97/DS4I_ASSIGNMENT_1/issues)



## Project Overview

This repository contains the code and documentation for Assignment 1 of the STA5073Z - Data Science for Industry 2025 course at the University of Cape Town.

The goal of this project is to construct and evaluate a neural network model that predicts the **Forecasted Avalanche Hazard (FAH)**, as published by the Scottish Avalanche Information Service (SAIS). We utilize a 15-year archive of avalanche forecasts, combining data on topography, weather conditions, and snow pack integrity.

- **Problem:** Multi-class classification (ordinal)
- **Target Variable:** `FAH` (Forecasted Avalanche Hazard)
- **Key Predictors:** Topography, Weather, Snow Pack Test results
- **Model:** Artificial Neural Network (ANN) built with TensorFlow/Keras

**Live Website:** https://saadz97.github.io/DS4I_ASSIGNMENT_1/   
**Final Report:** https://github.com/saadz97/DS4I_ASSIGNMENT_1/tree/main/Final%20Submission%20Files

## Project Structure

├── data/


│ ├── scotland_avalanche_forecasts_2009_2025.csv | **Original data (provided by course) - .gitignored**


│ └── training_data.RData, testing_data.RData | **Cleaned and preprocessed datasets - .gitignored**


├── qmd | **Quarto documents for analysis workflow**


│ ├── index.qmd | **Main scientific paper**


│ ├── Final_EDA.Rmd | **Cleaning Data**


│ ├── llm-reflection.qmd | **LLM usage documentation**


│ ├── tuning | **Hyperparameter tuning results**


│ └── model_dt.qmd | **Modelling and Results**


├── docs | **Rendered Quarto website for GitHub Pages**


├── Final Submission Files | **The Final Report**


├── _freeze | **Quarto freeze directory (cache)**



├── _quarto.yml | **Quarto project configuration**


├── README.md | **This file**


└── Assignment 1.Rproj | **RStudio Project file**


## Group Members

| Name | Student Number | Key Contributions |
| :--- | :--- | :--- |
| Ditiro Letsoalo | LTSDIT002 |Model development assistance, report writing and structuring of relevant sections, website help, README documentation |
| Saadia Abdullah | ABDSAA004 |  Setting up GitHub repository, Data cleaning, and EDA, including the write up for the relevant sections. |
| Ezra Goliath | GLTEZR001 | Model building, Model tuning, and Model evaluation, including the write up for relevant sections. |
| Hope Hennessy | HNNHOP001 | Data cleaning, EDA, and Website development, including the write up for the relevant sections. |
| Rachel Calaz | CLZRAC001 | Website development, write up for the abstract, introduction, literature review, discussion, and conclusion. |

## 🔑 Key Findings

- **Weather variables are most predictive**: Predictor Set 2 (weather conditions) achieved the highest validation accuracy (66%), slightly outperforming the comprehensive feature set
- **Optimal architecture**: 4 hidden layers with [30, 40, 40, 40] nodes provided the best balance of performance and complexity  
- **Class imbalance challenge**: Model excels at predicting Low hazard (85% sensitivity) but struggles with High hazard categories (6% sensitivity)
- **Overall performance**: 51.7% test accuracy, significantly better than random guessing (20%) but limited by severe class imbalance
- **Feature importance**: Weather data dominates predictive power, with topographic and snow-pack variables providing marginal incremental value
- **Practical implication**: Neural networks show promise as decision support tools but require human oversight, especially for high-consequence hazard levels

**Best Model**: 64.9% validation accuracy using all available predictors with careful hyperparameter optimization.

## How to Reproduce the Analysis

### 1. Clone the Repository
```bash
git clone https://github.com/saadz97/DS4I_ASSIGNMENT_1.git
cd DS4I_ASSIGNMENT_1

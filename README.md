# Predicting Forecasted Avalanche Hazard with Neural Networks

[![License](https://img.shields.io/github/license/saadz97/DS4I_ASSIGNMENT_1)](https://github.com/saadz97/DS4I_ASSIGNMENT_1/blob/main/LICENSE)
[![Last Commit](https://img.shields.io/github/last-commit/saadz97/DS4I_ASSIGNMENT_1)](https://github.com/saadz97/DS4I_ASSIGNMENT_1/commits/main)
[![Issues](https://img.shields.io/github/issues/saadz97/DS4I_ASSIGNMENT_1)](https://github.com/saadz97/DS4I_ASSIGNMENT_1/issues)



## Project Overview

This repository contains the code and documentation for Assignment 1 of the STA5073Z - Data Science for Industry 2025 course at the University of Cape Town.

The goal of this project is to construct and evaluate a neural network model that predicts the **Forecasted Avalanche Hazard (FAH)**, as published by the Scottish Avalanche Information Service (SAIS). We utilize a 15-year archive of avalanche forecasts, combining data on topography, weather conditions, and snow pack integrity.

- **Problem:** Multi-class classification (ordinal)
- **Target Variable:** `FAH` (Forecasted Avalanche Hazard)
- **Key Predictors:** Topography, Weather, Snow Pack Test results
- **Model:** Artificial Neural Network (ANN) built with TensorFlow/Keras

**Live Website:** https://github.com/saadz97/DS4I_ASSIGNMENT_1  
**Final Report:** [Direct link to the report on your GitHub Pages site]

## Project Structure

├── data/


│ ├── raw/ # Original data (provided by course) - .gitignored


│ └── processed/ # Cleaned and preprocessed datasets - .gitignored


├── R/ # R scripts for the analysis workflow


│ ├── 01_data_cleaning.R


│ ├── 02_eda.R


│ ├── 03_preprocessing.R


│ └── 04_modeling.R


├── docs/ # Rendered Quarto website for GitHub Pages


├── figures/ # Generated plots and visuals


├── _freeze/ # Quarto freeze directory (cache)


├── _output.yml # Quarto output configuration


├── _quarto.yml # Quarto project configuration


├── README.md # This file


├── requirements.R # R package dependencies


├── report.qmd # Main Quarto document for the scientific report


└── avalanche_forecasting.Rproj # RStudio Project file (optional but recommended)



## Group Members

| Name | Student Number | Key Contributions |
| :--- | :--- | :--- |
| [Group Member 1 Name] | [Student Number] | Data preprocessing, EDA, Model development |
| [Group Member 2 Name] | [Student Number] | Feature engineering, Hyperparameter tuning, LLM management |
| [Group Member 3 Name] | [Student Number] | Quarto website development, Writing, Visualization |
| [Group Member 4 Name] | [Student Number] | Presentation, Model evaluation, Report writing |
| [Group Member 4 Name] | [Student Number] | Presentation, Model evaluation, Report writing |

## Key Findings

Our analysis yielded the following key results:
- The best-performing model was a neural network with [X] hidden layers and [Y] units, achieving a test set accuracy of **[Z]%** and a Mean Absolute Error (MAE) of **[A]**.
- The most important predictors for the FAH were found to be `[Predictor 1]`, `[Predictor 2]`, and `[Predictor 3]`.
- The model demonstrates that neural networks can effectively learn the complex relationships used by expert forecasters at SAIS.

## How to Reproduce the Analysis

### 1. Clone the Repository
```bash
git clone https://github.com/saadz97/DS4I_ASSIGNMENT_1.git
cd DS4I_ASSIGNMENT_1

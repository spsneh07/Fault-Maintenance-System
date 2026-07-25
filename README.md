# Bearing Fault Diagnosis System

This repository contains data analysis, feature engineering, and deep learning models to diagnose bearing faults using vibration signals. The project focuses on classifying the operating state of bearings into multiple conditions such as Normal (OK), Inner Race (IR) fault, Outer Race (OR) fault, and Rolling Element (RE) fault.

## Project Structure

- **Notebook & Reports:** 
  - `notebook.ipynb`: The main Jupyter Notebook containing the full methodology, data exploration, preprocessing, and modeling.
  - `NOTEBOOK_METHODOLOGY_REPORT.md` / `notebook report.pdf`: Detailed methodology and results.
  - `ann.html` / `deep learning.html`: Interactive HTML exports of model execution and reports.
- **Deep Learning Models:**
  - `run_ann.py` / `ann_training.py`: Artificial Neural Network (MLP) implementation.
  - `run_cnn.py`: 1D Convolutional Neural Network implementation.
  - `run_lstm.py`: Long Short-Term Memory implementation.
  - `run_rnn.py`: Recurrent Neural Network implementation.
- **Data & Features:**
  - `bearing_features.csv` / `bearing_features_phase3.csv`: Extracted time-domain and frequency-domain features from the raw vibration data.
  - `Similar System Bearing Data Set Description.pdf`: Documentation regarding the dataset.
- **Dimensionality Reduction:**
  - `pca_mini_project.py`: Principal Component Analysis (PCA) used for feature dimensionality reduction, accompanied by scatter plots (`pca_biplot.png`, `pca_scree.png`).

## Setup and Usage

1. **Environment Setup:** It is recommended to use a virtual environment (`.venv`). Install dependencies like `tensorflow`, `scikit-learn`, `pandas`, `numpy`, `matplotlib`, and `seaborn`.
2. **Feature Extraction:** The raw dataset is processed to extract key features (mean, std, kurtosis, etc.) saved into the CSV files.
3. **Training:** You can run any of the standalone `run_*.py` scripts to train the respective models on the extracted feature datasets.

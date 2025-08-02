# Heart Disease Prediction

This repository contains a machine learning project that predicts the likelihood of heart disease using the UCI Heart Disease dataset. The project includes a Jupyter notebook with comprehensive analysis and a Streamlit web application for interactive predictions using a Support Vector Machine (SVM) model.

## Project Overview

- **Objective**: Develop a binary classification model to predict heart disease risk based on patient data.
- **Dataset**: UCI Heart Disease dataset (920 records), preprocessed to handle missing values, outliers, and categorical variables.
- **Model**: SVM achieving 88.96% accuracy and 90.43% F1-score.
- **Tools**: Python, pandas, scikit-learn, Seaborn, Matplotlib, Streamlit.

## Features

- **Data Preprocessing**: Null value imputation (median/mode), outlier handling (IQR), one-hot encoding, and robust scaling.
- **Model Training**: Trained and evaluated multiple models (Logistic Regression, Random Forest, XGBoost, SVM).
- **Visualization**: Exploratory data analysis with histograms, correlation heatmaps, and feature importance plots.
- **Web Application**: Interactive Streamlit app for real-time heart disease risk prediction with user input sliders and selectboxes.

## Live Demo

Check out the deployed Streamlit app [HERE](https://ml-heart-disease-predict.streamlit.app/)

## Notebook and Analysis

The detailed analysis, including data preprocessing, model training, and visualizations, is available in this Kaggle notebook:  
[Kaggle Notebook](https://www.kaggle.com/code/advay1235/heart-disease-prediction-analysis)

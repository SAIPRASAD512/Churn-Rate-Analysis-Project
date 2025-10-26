# Customer Churn Prediction Project

## Table of Contents
- [Project Overview](#project-overview)
- [Repository Structure](#repository-structure)
- [Features](#features)
- [Setup Instructions](#setup-instructions)
- [Usage](#usage)
- [Power BI Dashboard](#power-bi-dashboard)
- [Authors](#authors)

---

## Project Overview
This project aims to predict customer churn using machine learning and provide actionable insights via a Power BI dashboard. The workflow involves:

1. Data extraction and cleaning – SQL scripts to clean and prepare customer data.
2. Exploratory Data Analysis (EDA) – Python scripts for understanding data distributions and correlations.
3. Machine Learning Model – A Random Forest-based classifier trained to predict churn.
4. Dashboard & Reporting – Power BI dashboards to visualize churn trends and insights.

This project provides a complete end-to-end pipeline from raw data to actionable insights.

---

## Repository Structure

- **Churn Prediction**
  - churn_etl_eda.py – Data preprocessing, EDA, and feature engineering
  - churn_model.py – Model training, evaluation, hyperparameter tuning
  - churn_rate_predict_001.ipynb – Jupyter notebook showcasing predictions and workflow
  - encoders.pkl – Saved LabelEncoders for categorical features
  - random_forest_model.pkl – Trained Random Forest model

- **Data**
  - Customer_Data.csv – Raw customer data used for analysis and model training

- **Power BI**
  - Churn Analysis.pdf – Exported PDF report of the dashboard
  - Churn Analysis-1.png – Full dashboard screenshot
  - Churn Analysis-2.png – Tooltip / detailed view screenshot

- **SQL**
  - 01_create_db.sql – Script to create database and initial tables
  - 02_data_cleanup.sql – Script to clean and preprocess data
  - 03_creating_prod_churn.sql – Script to create production-ready churn dataset & views

- requirements.txt – Python dependencies

---

## Features

- **Data Cleaning & EDA**
  - Handles missing values and inconsistent entries in the dataset.
  - Provides visualizations for numerical and categorical features.
  - Computes correlations and distributions for insights.

- **Modeling**
  - Random Forest classifier for churn prediction.
  - Handles class imbalance using SMOTE.
  - Cross-validation and hyperparameter tuning for optimal performance.
  - Pickled models for reuse.

- **Power BI Dashboard**
  - Provides visual analysis of churn trends, customer demographics, and service usage.
  - Exported as PDF and screenshots for reference.

- **SQL Pipeline**
  - Database creation, data cleanup, and production-ready dataset creation using SQL scripts.

---

## Setup Instructions

1. Clone the repository.
2. Install dependencies using the provided requirements.txt file.
3. Prepare the database using the SQL scripts in the following order:
   - 01__create__db.sql
   - 02__data__cleanup.sql
   - 03__creating__prod__churn.sql
4. Run Python scripts for EDA and model training.
5. Use the Power BI dashboard files or PDF for visualization and reporting.

---

## Usage

- The project allows predicting churn for new customers using the trained Random Forest model.
- Encoders are provided to transform categorical features for model compatibility.
- EDA scripts provide insights into the data distributions and correlations.

---

## Authors
- Your Name – [GitHub Profile](https://github.com/SAIPRASAD512)


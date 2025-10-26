# Purpose: Load data, clean it, perform EDA, and save encoders.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import pickle

# Load Data
df = pd.read_csv("Customer_Churn_Data.csv")

# Drop CustomerID
df = df.drop(columns=["customerID"])

# Fix TotalCharges
df["TotalCharges"] = df["TotalCharges"].replace(' ', 0).astype(float)

# Target Distribution
print("Churn distribution:\n", df["Churn"].value_counts())

# EDA: Numerical Features
numerical_features = ['tenure', 'MonthlyCharges', 'TotalCharges']

def plot_histogram(df, column):
    plt.figure(figsize=(5,3))
    sns.histplot(df[column], kde=True)
    plt.axvline(df[column].mean(), color='r', linestyle='--', label='Mean')
    plt.axvline(df[column].median(), color='g', linestyle='-', label='Median')
    plt.title(f'Distribution of {column}')
    plt.legend()
    plt.show()

def plot_box(df, column):
    plt.figure(figsize=(5,3))
    sns.boxplot(y=df[column])
    plt.title(f'Box plot of {column}')
    plt.show()

for col in numerical_features:
    plot_histogram(df, col)
    plot_box(df, col)

# Correlation heatmap
plt.figure(figsize=(8,4))
sns.heatmap(df[numerical_features].corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix")
plt.show()

# Categorical Features
object_cols = df.select_dtypes(include='object').columns.tolist()
object_cols = ["SeniorCitizen"] + object_cols

for col in object_cols:
    plt.figure(figsize=(5,3))
    sns.countplot(x=df[col])
    plt.title(f'Count plot of {col}')
    plt.show()

# Data Preprocessing- Encode target
df["Churn"] = df["Churn"].replace({"No": 0, "Yes": 1})

# Label encode categorical features
encoders = {}
for col in object_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# Save encoders
with open("encoders.pkl", "wb") as f:
    pickle.dump(encoders, f)

# Save cleaned dataframe
df.to_csv("cleaned_customer_churn.csv", index=False)
print("ETL and EDA completed. Data saved to cleaned_customer_churn.csv")

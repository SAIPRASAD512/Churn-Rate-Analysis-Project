# Purpose: Train models, evaluate, tune hyperparameters, and make predictions.

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle

# Load cleaned data
df = pd.read_csv("cleaned_customer_churn.csv")

# Features and target
X = df.drop(columns=['Churn'])
y = df['Churn']

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Handle Imbalanced Data
smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

# Model Training
models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "XGBoost": XGBClassifier(random_state=42)
}

cv_scores = {}
for name, model in models.items():
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X_train_sm, y_train_sm, cv=skf, scoring='accuracy')
    cv_scores[name] = scores
    print(f"{name} CV Accuracy: {np.mean(scores):.2f} ± {np.std(scores):.2f}")

# Train best model (Random Forest)
rfc = RandomForestClassifier(random_state=42)
rfc.fit(X_train_sm, y_train_sm)

# Save model
model_data = {"model": rfc, "features": X.columns.tolist()}
with open("random_forest_model.pkl", "wb") as f:
    pickle.dump(model_data, f)

# Evaluate
y_pred = rfc.predict(X_test)
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Hyperparameter Tuning
param_grid = {
    'n_estimators': [100, 150],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 4],
    'min_samples_leaf': [1, 2],
    'max_features': ['sqrt'],
    'class_weight': ['balanced']
}

grid = GridSearchCV(
    estimator=rfc,
    param_grid=param_grid,
    scoring='f1',
    cv=5,
    n_jobs=-1,
    verbose=2
)
grid.fit(X_train_sm, y_train_sm)
best_rfc = grid.best_estimator_
print("Best Parameters:", grid.best_params_)

# Evaluate tuned model
y_pred_best = best_rfc.predict(X_test)
print("Tuned Model Accuracy:", accuracy_score(y_test, y_pred_best))
print(confusion_matrix(y_test, y_pred_best))
print(classification_report(y_test, y_pred_best))

# Example Prediction
customer_data = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 1,
    "PhoneService": "No",
    "MultipleLines": "No phone service",
    "InternetService": "DSL",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 29.85,
    "TotalCharges": 29.85,
}

input_df = pd.DataFrame([customer_data])

# Load encoders
with open("encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

# Encode categorical columns
for col, enc in encoders.items():
    if col in input_df.columns:
        input_df[col] = enc.transform(input_df[col])

# Predict
prediction = best_rfc.predict(input_df)
prediction_proba = best_rfc.predict_proba(input_df)
print(f"Prediction: {'Churn' if prediction[0]==1 else 'No Churn'}")
print(f"Probability of Churn: {prediction_proba[0][1]:.2f}")

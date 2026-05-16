import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.datasets import load_diabetes
import mlflow
import mlflow.sklearn
import joblib
import os
from datetime import datetime

os.makedirs("data", exist_ok=True)
os.makedirs("model", exist_ok=True)

print("🚀 Starting Diabetes MLOps Training Pipeline...")

# ====================== LOAD DIABETES DATASET ======================
from sklearn.datasets import load_breast_cancer
raw = load_breast_cancer()
data = pd.DataFrame(raw.data, columns=raw.feature_names)
data["target"] = raw.target
data.to_csv("data/dataset.csv", index=False)
print(f"✅ Diabetes Dataset loaded: {data.shape[0]} rows, {data.shape[1]} columns")

# ====================== MLFLOW SETUP ======================
mlflow.set_experiment("mlops-diabetes-prediction")

with mlflow.start_run(run_name=f"run_{datetime.now().strftime('%Y%m%d_%H%M')}") as run:

    mlflow.log_param("model_type", "RandomForestClassifier")
    mlflow.log_param("n_estimators", 200)
    mlflow.log_param("random_state", 42)
    mlflow.log_param("test_size", 0.2)
    mlflow.log_param("dataset", "sklearn_breast_cancer")

    X = data.drop("target", axis=1)
    y = data["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)

    joblib.dump(model, "model/model.pkl")
    mlflow.sklearn.log_model(model, "random_forest_model")

    print("\n✅ Model Training Completed!")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"Run ID   : {run.info.run_id}")
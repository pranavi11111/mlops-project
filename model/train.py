import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
import mlflow.sklearn
import joblib
import os
from datetime import datetime

# Create directories
os.makedirs("data", exist_ok=True)
os.makedirs("model", exist_ok=True)

print("🚀 Starting MLOps Training Pipeline...")

# ====================== DATA GENERATION ======================
np.random.seed(42)
n = 5000  # Increased dataset size

data = pd.DataFrame({
    "age": np.random.randint(18, 80, n),
    "income": np.random.randint(15000, 150000, n),
    "education": np.random.randint(1, 6, n),      # 1=High School to 5=PhD
    "experience": np.random.randint(0, 40, n),
    "survived": np.random.randint(0, 2, n)
})

data.to_csv("data/dataset.csv", index=False)
print(f"✅ Dataset created: {data.shape[0]} rows")

# ====================== MLFLOW SETUP ======================
mlflow.set_experiment("mlops-titanic-survival")

with mlflow.start_run(run_name=f"run_{datetime.now().strftime('%Y%m%d_%H%M')}") as run:
    
    # Log parameters
    mlflow.log_param("model_type", "RandomForestClassifier")
    mlflow.log_param("n_estimators", 200)
    mlflow.log_param("random_state", 42)
    mlflow.log_param("test_size", 0.2)
    
    # Split data
    X = data.drop("survived", axis=1)
    y = data["survived"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    # Predict & Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Log metrics
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)

    # Save model locally
    joblib.dump(model, "model/model.pkl")
    
    # Log model to MLflow
    mlflow.sklearn.log_model(model, "random_forest_model")

    print("\n✅ Model Training Completed!")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"Run ID   : {run.info.run_id}")
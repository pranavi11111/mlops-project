import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
import mlflow.sklearn
import joblib
import os

# Create data folder if not exists
os.makedirs("data", exist_ok=True)
os.makedirs("model", exist_ok=True)

# Generate simple dataset (Titanic-like)
np.random.seed(42)
n = 1000

data = pd.DataFrame({
    "age": np.random.randint(1, 80, n),
    "income": np.random.randint(20000, 120000, n),
    "education": np.random.randint(1, 5, n),
    "experience": np.random.randint(0, 40, n),
    "survived": np.random.randint(0, 2, n)
})

# Save dataset
data.to_csv("data/dataset.csv", index=False)
print("✅ Dataset created and saved!")

# Split data
X = data.drop("survived", axis=1)
y = data["survived"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# MLflow tracking
mlflow.set_experiment("mlops-experiment")

with mlflow.start_run():
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Log metrics
    mlflow.log_param("n_estimators", 100)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)

    # Save model
    joblib.dump(model, "model/model.pkl")
    mlflow.sklearn.log_model(model, "model")

    print(f"✅ Model trained!")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
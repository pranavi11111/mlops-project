import pandas as pd
import numpy as np
from evidently.dashboard import Dashboard
from evidently.tabs import DataDriftTab
import os

os.makedirs("monitoring/reports", exist_ok=True)

reference = pd.DataFrame({
    "Pregnancies": np.random.randint(0, 10, 500),
    "Glucose": np.random.randint(70, 140, 500),
    "BloodPressure": np.random.randint(60, 90, 500),
    "BMI": np.random.uniform(20, 40, 500),
    "Age": np.random.randint(20, 60, 500),
})

current = pd.DataFrame({
    "Pregnancies": np.random.randint(2, 12, 200),
    "Glucose": np.random.randint(100, 180, 200),
    "BloodPressure": np.random.randint(65, 95, 200),
    "BMI": np.random.uniform(25, 45, 200),
    "Age": np.random.randint(30, 70, 200),
})

dashboard = Dashboard(tabs=[DataDriftTab()])
dashboard.calculate(reference, current)
dashboard.save("monitoring/reports/evidently_report.html")
print("Evidently report saved!")
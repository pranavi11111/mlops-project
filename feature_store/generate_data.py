import pandas as pd
from datetime import datetime
import os

os.makedirs("data", exist_ok=True)

df = pd.DataFrame({
    "patient_id": [1, 2, 3, 4, 5],
    "mean_radius": [17.99, 13.54, 20.57, 11.42, 15.78],
    "mean_texture": [10.38, 14.36, 17.77, 20.38, 17.89],
    "mean_perimeter": [122.8, 87.46, 132.9, 77.58, 103.6],
    "mean_area": [1001.0, 566.3, 1326.0, 386.1, 781.0],
    "mean_smoothness": [0.1184, 0.09779, 0.08474, 0.1425, 0.0971],
    "mean_concavity": [0.3001, 0.06664, 0.1986, 0.2439, 0.1752],
    "event_timestamp": [datetime(2024, 1, 1)] * 5
})

df.to_parquet("data/patient_features.parquet", index=False)
print("✅ Feature data created!")
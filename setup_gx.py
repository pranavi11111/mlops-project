import great_expectations as gx
import pandas as pd
import os

print("🚀 Setting up Great Expectations for MLOps...")

context = gx.get_context(mode="file", project_root_dir="./gx")

print("✅ Great Expectations initialized successfully!")
print(f"GX Directory: {context.root_directory}")

if os.path.exists("data/dataset.csv"):
    df = pd.read_csv("data/dataset.csv")
    print(f"✅ Dataset loaded: {df.shape[0]} rows")
    print(f"Columns: {list(df.columns)}")
else:
    print("⚠️ Dataset not found. Run python model/train.py first")

print("\n🎉 Setup Complete! You can now create Expectations.")
import shap
import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

os.makedirs("monitoring/reports", exist_ok=True)

print("Loading model...")
model = joblib.load("model/model.pkl")

feature_names = [
    "Pregnancies", "Glucose", "BloodPressure",
    "SkinThickness", "Insulin", "BMI",
    "DiabetesPedigreeFunction", "Age"
]

patient = pd.DataFrame([[
    6, 148, 72, 35, 0, 33.6, 0.627, 50
]], columns=feature_names)

print("Generating SHAP explanation...")

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(patient)

prediction = model.predict(patient)[0]
confidence = max(model.predict_proba(patient)[0])
result = "DIABETIC" if prediction == 1 else "NOT DIABETIC"

print("\n" + "="*50)
print("PATIENT ANALYSIS REPORT")
print("="*50)
print(f"Prediction: {result}")
print(f"Confidence: {confidence:.1%}")
print("\nFeature Impact (Why this prediction?):")
print("-"*50)

if isinstance(shap_values, list):
    sv = shap_values[1][0]
else:
    sv = shap_values[0]

sv_list = np.array(sv).flatten().tolist()
val_list = patient.values[0].tolist()

combined = list(zip(feature_names, sv_list, val_list))
combined.sort(key=lambda x: abs(x[1]), reverse=True)

for feat, impact, value in combined:
    if impact > 0.01:
        emoji = "🔴"
        direction = "increases diabetes risk"
    elif impact < -0.01:
        emoji = "🟢"
        direction = "decreases diabetes risk"
    else:
        emoji = "🟡"
        direction = "neutral"
    print(f"{emoji} {feat}: {value} → impact: {impact:+.3f} ({direction})")

print("="*50)

# Save plot
try:
    plt.figure(figsize=(10, 6))
    if isinstance(shap_values, list):
        sv_plot = shap_values[1]
    else:
        sv_plot = shap_values
    shap.summary_plot(
        sv_plot, patient,
        feature_names=feature_names,
        plot_type="bar", show=False
    )
    plt.title(f"SHAP Feature Importance - {result}")
    plt.tight_layout()
    plt.savefig("monitoring/reports/shap_explanation.png",
                dpi=150, bbox_inches='tight')
    plt.close()
    print("SHAP plot saved!")
except Exception as e:
    print(f"Plot error (non-critical): {e}")

# HTML Report
html = f"""<!DOCTYPE html>
<html>
<head>
    <title>XAI Explanation Report</title>
    <style>
        body {{ font-family: Arial; background: #0f172a;
               color: white; padding: 30px; }}
        h1 {{ color: #60a5fa; }}
        .card {{ background: #1e293b; border-radius: 10px;
                padding: 20px; margin: 15px 0; }}
        .diabetic {{ border-left: 5px solid #ef4444; }}
        .healthy {{ border-left: 5px solid #22c55e; }}
        .badge-red {{ background: #ef4444; padding: 5px 15px;
                     border-radius: 5px; font-weight: bold; }}
        .badge-green {{ background: #22c55e; padding: 5px 15px;
                       border-radius: 5px; font-weight: bold; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th {{ background: #334155; padding: 12px; text-align: left; }}
        td {{ padding: 12px; border-bottom: 1px solid #334155; }}
        .high {{ color: #ef4444; font-weight: bold; }}
        .low {{ color: #22c55e; font-weight: bold; }}
        .neutral {{ color: #facc15; }}
        img {{ width: 100%; border-radius: 10px; margin-top: 15px; }}
    </style>
</head>
<body>
    <h1>Explainable AI - Patient Diagnosis</h1>
    <div class="card {'diabetic' if prediction == 1 else 'healthy'}">
        <h2>Result</h2>
        <p>Diagnosis:
           <span class="{'badge-red' if prediction==1 else 'badge-green'}">
           {result}
           </span>
        </p>
        <p>Confidence: <b>{confidence:.1%}</b></p>
    </div>
    <div class="card">
        <h2>Feature Impact Analysis</h2>
        <table>
            <tr>
                <th>Feature</th><th>Value</th>
                <th>SHAP Impact</th><th>Risk Level</th>
            </tr>"""

for feat, impact, value in combined:
    if impact > 0.01:
        risk = '<span class="high">HIGH RISK</span>'
    elif impact < -0.01:
        risk = '<span class="low">PROTECTIVE</span>'
    else:
        risk = '<span class="neutral">NEUTRAL</span>'
    html += f"""
            <tr>
                <td>{feat}</td><td>{value}</td>
                <td>{impact:+.3f}</td><td>{risk}</td>
            </tr>"""

html += """
        </table>
        <img src="shap_explanation.png" alt="SHAP Chart"/>
    </div>
    <div class="card">
        <h2>What is SHAP?</h2>
        <p>SHAP values show exactly how much each feature
        contributed to the prediction. Positive values
        increase diabetes risk, negative values decrease it.
        </p>
    </div>
</body>
</html>"""

with open("monitoring/reports/xai_report.html",
          "w", encoding="utf-8") as f:
    f.write(html)

print("XAI HTML report saved!")
print("Open: monitoring/reports/xai_report.html")
print("ALL DONE!")
import pandas as pd
import numpy as np
import json
import os

os.makedirs("monitoring/reports", exist_ok=True)

print("Generating drift report...")

# Reference data
reference_data = pd.DataFrame({
    "age": np.random.randint(1, 80, 500),
    "income": np.random.randint(20000, 120000, 500),
    "education": np.random.randint(1, 5, 500),
    "experience": np.random.randint(0, 40, 500),
})

# Current data (drifted)
current_data = pd.DataFrame({
    "age": np.random.randint(25, 60, 200),
    "income": np.random.randint(50000, 150000, 200),
    "education": np.random.randint(2, 5, 200),
    "experience": np.random.randint(5, 40, 200),
})

# Calculate drift manually
drift_results = {}
for col in reference_data.columns:
    ref_mean = reference_data[col].mean()
    cur_mean = current_data[col].mean()
    drift_score = abs(ref_mean - cur_mean) / ref_mean
    drift_results[col] = {
        "reference_mean": round(ref_mean, 2),
        "current_mean": round(cur_mean, 2),
        "drift_score": round(drift_score, 4),
        "drifted": bool(drift_score > 0.1)
    }

# Generate HTML report
html = """<!DOCTYPE html>
<html>
<head>
    <title>MLOps Drift Report</title>
    <style>
        body { font-family: Arial; background: #0f172a; color: white; padding: 30px; }
        h1 { color: #60a5fa; }
        h2 { color: #94a3b8; }
        .card { background: #1e293b; border-radius: 10px; padding: 20px; margin: 15px 0; }
        .badge-red { background: #ef4444; padding: 3px 10px; border-radius: 5px; font-size: 12px; }
        .badge-green { background: #22c55e; padding: 3px 10px; border-radius: 5px; font-size: 12px; }
        table { width: 100%; border-collapse: collapse; }
        th { background: #334155; padding: 10px; text-align: left; }
        td { padding: 10px; border-bottom: 1px solid #334155; }
    </style>
</head>
<body>
    <h1>MLOps Data Drift Report</h1>
    <h2>Drift Monitor</h2>
    <div class="card">
        <h2>Drift Summary</h2>
        <table>
            <tr>
                <th>Feature</th>
                <th>Reference Mean</th>
                <th>Current Mean</th>
                <th>Drift Score</th>
                <th>Status</th>
            </tr>
"""

for col, result in drift_results.items():
    if result["drifted"]:
        status = '<span class="badge-red">DRIFTED</span>'
    else:
        status = '<span class="badge-green">OK</span>'
    html += "<tr>"
    html += "<td>" + col + "</td>"
    html += "<td>" + str(result["reference_mean"]) + "</td>"
    html += "<td>" + str(result["current_mean"]) + "</td>"
    html += "<td>" + str(result["drift_score"]) + "</td>"
    html += "<td>" + status + "</td>"
    html += "</tr>"

html += """
        </table>
    </div>
    <div class="card">
        <h2>Conclusion</h2>
        <p>Data drift detected in production data compared to training data.
        Model retraining recommended if drift score exceeds 0.1 threshold.</p>
    </div>
</body>
</html>"""

with open("monitoring/reports/drift_report.html", "w", encoding="utf-8") as f:
    f.write(html)

print("Drift report generated!")
print("Open: monitoring/reports/drift_report.html")

with open("monitoring/reports/drift_results.json", "w", encoding="utf-8") as f:
    json.dump(drift_results, f, indent=2, default=str)

print("JSON results saved!")
print("All done!")
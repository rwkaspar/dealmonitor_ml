import os, sys, json
from datetime import datetime
from jinja2 import Template
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.features import clean_price_user


# 🔍 Load latest prediction file
pred_dir = "predictions"
latest = sorted(os.listdir(pred_dir))[-1]
path = os.path.join(pred_dir, latest)

print(f"🧾 Generating report for {path}...")

# Load predictions
with open(path) as f:
    data = [json.loads(line) for line in f]

# Clean & format
for row in data:
    row["price_user_clean"] = clean_price_user(row.get("price_user"))
    row["is_hit"] = abs((row.get("price_user_clean") or 0) - (row.get("predicted_price") or -999)) < 0.01

    top = row.get("top_candidates", [])[:3]
    row["top_html"] = []
    for val, score in top:
        match = abs((row.get("price_user_clean") or 0) - val) < 0.01
        row["top_html"].append({
            "val": f"{val:.2f}",
            "score": f"{score:.3f}",
            "match": match
        })

# HTML Template
template = Template("""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Prediction Report</title>
    <style>
        body { font-family: Arial, sans-serif; padding: 2em; }
        table { border-collapse: collapse; width: 100%; }
        th, td { border: 1px solid #ccc; padding: 8px; text-align: left; }
        .hit { background-color: #c8facc; }
        .miss { background-color: #faccd1; }
        .match { font-weight: bold; color: green; }
        .nomatch { color: #888; }
    </style>
</head>
<body>
    <h2>DealMonitor Prediction Report ({{ date }})</h2>
    <p>Total rows: {{ data|length }}</p>
    <table>
        <thead>
            <tr>
                <th>URL</th>
                <th>User Price</th>
                <th>Predicted</th>
                <th>Top 3 Candidates</th>
            </tr>
        </thead>
        <tbody>
        {% for row in data %}
            <tr class="{{ 'hit' if row.is_hit else 'miss' }}">
                <td>{{ row.url }}</td>
                <td>{{ row.price_user_clean }}</td>
                <td>{{ row.predicted_price }}</td>
                <td>
                    {% for cand in row.top_html %}
                        <div class="{{ 'match' if cand.match else 'nomatch' }}">
                            {{ cand.val }} ({{ cand.score }})
                        </div>
                    {% endfor %}
                </td>
            </tr>
        {% endfor %}
        </tbody>
    </table>
</body>
</html>
""")

# Render + Save
os.makedirs("reports", exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
out_path = f"reports/report_{timestamp}.html"

with open(out_path, "w") as f:
    f.write(template.render(data=data, date=timestamp))

print(f"✅ Report saved to {out_path}")

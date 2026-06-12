import streamlit as st
import pandas as pd
import json
from pathlib import Path

from src.config import LOG_FILE

st.set_page_config(page_title="Churn Monitoring Dashboard", layout="wide")

st.title("📊 Churn Prediction Monitoring Dashboard")

# -----------------------------
# Load logs
# -----------------------------
def load_logs():
    if not LOG_FILE.exists():
        return pd.DataFrame()

    records = []
    with open(LOG_FILE, "r") as f:
        for line in f:
            try:
                records.append(json.loads(line))
            except:
                pass

    return pd.DataFrame(records)

df = load_logs()

# -----------------------------
# Empty state
# -----------------------------
if df.empty:
    st.warning("No predictions logged yet.")
    st.stop()

# -----------------------------
# Metrics
# -----------------------------
total = len(df)
churn_rate = df["prediction"].mean()
avg_latency = df["latency"].mean()

col1, col2, col3 = st.columns(3)

col1.metric("Total Predictions", total)
col2.metric("Churn Rate", f"{churn_rate:.2%}")
col3.metric("Avg Latency (ms)", f"{avg_latency * 1000:.2f}")

# -----------------------------
# Debug (temporary)
# -----------------------------
st.subheader("Raw Data")
st.write(df.drop(columns=["input"], errors="ignore").head())

# -----------------------------
# Prediction distribution
# -----------------------------
st.subheader("Prediction Distribution")

chart_data = pd.DataFrame({
    "Prediction": ["No Churn", "Churn"],
    "Count": [
        (df["prediction"] == 0).sum(),
        (df["prediction"] == 1).sum()
    ]
})



st.write("Predictions:")
st.write(df["prediction"].value_counts(dropna=False))

chart_data.columns = ["prediction", "count"]

st.bar_chart(chart_data.set_index("prediction"))

# -----------------------------
# Latency
# -----------------------------
st.subheader("Latency Over Time")
st.line_chart(df["latency"])
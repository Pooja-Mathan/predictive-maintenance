"""Streamlit dashboard for predictive maintenance.

Run:  streamlit run app/app.py
"""
import json
import os
import sys

import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_preprocessing import RENAME_MAP  # noqa: E402
from src.predict import DEFAULT_MODEL, load_model, predict_df, predict_one  # noqa: E402

st.set_page_config(page_title="Predictive Maintenance", page_icon="🛠️", layout="centered")
st.title("🛠️ Predictive Maintenance")
st.caption("Predict machine failure from sensor readings (AI4I 2020 dataset).")


@st.cache_resource
def get_model():
    return load_model(DEFAULT_MODEL)


try:
    model = get_model()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

tab_single, tab_batch, tab_perf = st.tabs(["Single machine", "Batch CSV", "Model performance"])

with tab_single:
    c1, c2 = st.columns(2)
    type_ = c1.selectbox("Product type", ["L", "M", "H"], index=1)
    air = c1.number_input("Air temperature [K]", 295.0, 305.0, 300.0, 0.1)
    proc = c1.number_input("Process temperature [K]", 305.0, 315.0, 310.0, 0.1)
    rpm = c2.number_input("Rotational speed [rpm]", 1000, 3000, 1500, 10)
    torque = c2.number_input("Torque [Nm]", 3.0, 80.0, 40.0, 0.1)
    wear = c2.number_input("Tool wear [min]", 0, 300, 100, 1)

    if st.button("Predict", type="primary"):
        pred, prob = predict_one(model, type_, air, proc, rpm, torque, wear)
        st.metric("Failure probability", f"{prob:.1%}")
        st.progress(min(prob, 1.0))
        if pred:
            st.error("⚠️ High risk of failure — schedule maintenance.")
        else:
            st.success("✅ Machine looks healthy.")

with tab_batch:
    st.write("Upload a CSV with the raw AI4I columns "
             "(`Type`, `Air temperature [K]`, `Process temperature [K]`, "
             "`Rotational speed [rpm]`, `Torque [Nm]`, `Tool wear [min]`).")
    up = st.file_uploader("CSV file", type="csv")
    if up:
        df = pd.read_csv(up).rename(columns=RENAME_MAP)
        try:
            res = predict_df(model, df)
            st.dataframe(res.sort_values("failure_probability", ascending=False))
            st.download_button("Download predictions", res.to_csv(index=False),
                               "predictions.csv", "text/csv")
        except KeyError as e:
            st.error(f"Missing required column: {e}")

with tab_perf:
    mp = os.getenv("METRICS_PATH", "models/metrics.json")
    if os.path.exists(mp):
        with open(mp) as f:
            m = json.load(f)
        st.write(f"**Best model:** `{m['best_model']}`")
        st.dataframe(pd.DataFrame(m["results"]).drop(index="confusion_matrix").T.round(3))
    else:
        st.info("Run `python -m src.train` to generate metrics.")

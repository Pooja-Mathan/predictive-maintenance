"""Run failure predictions with a trained model.

Usage (single machine):
    python -m src.predict --type M --air-temp 300 --process-temp 310 \
        --rpm 1500 --torque 40 --tool-wear 100

Usage (batch CSV with the raw AI4I column names):
    python -m src.predict --csv data/new_readings.csv --out predictions.csv
"""
from __future__ import annotations

import argparse
import os

import joblib
import pandas as pd
from dotenv import load_dotenv

from src.data_preprocessing import FEATURES, RENAME_MAP, add_features

load_dotenv()
DEFAULT_MODEL = os.getenv("MODEL_PATH", "models/model.joblib")


def load_model(path: str = DEFAULT_MODEL):
    try:
        return joblib.load(path)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Model not found at '{path}'. Run `python -m src.train` first.") from e


def predict_df(model, df: pd.DataFrame) -> pd.DataFrame:
    """Accepts clean column names (type, air_temp, ...) and returns df + probability + prediction."""
    X = add_features(df)[FEATURES]
    out = df.copy()
    out["failure_probability"] = model.predict_proba(X)[:, 1]
    out["prediction"] = (out["failure_probability"] >= 0.5).astype(int)
    return out


def predict_one(model, type_: str, air_temp: float, process_temp: float,
                rpm: float, torque: float, tool_wear: float) -> tuple[int, float]:
    row = pd.DataFrame([{
        "type": type_, "air_temp": air_temp, "process_temp": process_temp,
        "rpm": rpm, "torque": torque, "tool_wear": tool_wear,
    }])
    res = predict_df(model, row).iloc[0]
    return int(res["prediction"]), float(res["failure_probability"])


def main() -> None:
    p = argparse.ArgumentParser(description="Predict machine failure")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--csv", help="Batch input CSV (raw AI4I column names)")
    p.add_argument("--out", default="predictions.csv")
    p.add_argument("--type", choices=["L", "M", "H"])
    p.add_argument("--air-temp", type=float)
    p.add_argument("--process-temp", type=float)
    p.add_argument("--rpm", type=float)
    p.add_argument("--torque", type=float)
    p.add_argument("--tool-wear", type=float)
    a = p.parse_args()

    model = load_model(a.model)

    if a.csv:
        df = pd.read_csv(a.csv).rename(columns=RENAME_MAP)
        res = predict_df(model, df)
        res.to_csv(a.out, index=False)
        print(f"Wrote {len(res)} predictions -> {a.out}")
        return

    needed = [a.type, a.air_temp, a.process_temp, a.rpm, a.torque, a.tool_wear]
    if any(v is None for v in needed):
        p.error("Provide --csv, or all of: --type --air-temp --process-temp --rpm --torque --tool-wear")

    pred, prob = predict_one(model, *needed)
    print(f"Prediction: {'FAILURE' if pred else 'OK'} (probability of failure = {prob:.1%})")


if __name__ == "__main__":
    main()

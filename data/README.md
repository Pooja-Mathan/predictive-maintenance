# Data

This project uses the **AI4I 2020 Predictive Maintenance Dataset** (10,000 rows, synthetic).

## Download

1. Go to <https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset>
2. Download the CSV and save it as **`data/ai4i2020.csv`**.

(It is also mirrored on Kaggle as "Machine Predictive Maintenance Classification" / "AI4I 2020".)

## Columns

| Column | Description |
|---|---|
| UDI | Row id (dropped) |
| Product ID | Quality letter + serial number (dropped) |
| Type | Product quality variant: L (50%), M (30%), H (20%) |
| Air temperature [K] | Ambient temperature |
| Process temperature [K] | Process temperature |
| Rotational speed [rpm] | Spindle speed |
| Torque [Nm] | Torque |
| Tool wear [min] | Accumulated tool wear |
| Machine failure | **Target**: 1 if any failure mode occurred |
| TWF, HDF, PWF, OSF, RNF | Individual failure modes (dropped to avoid target leakage) |

The CSV is git-ignored, so it is not committed to the repo.

# 🛠️ Predictive Maintenance — AI4I 2020

Machine-learning project that predicts **machine failure** from sensor readings using the
[AI4I 2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset)
(10,000 records, ~3.4% failures).

Includes an EDA/modeling notebook, a reproducible training pipeline, a CLI for predictions, and a Streamlit web app.

## Project structure

```
predictive-maintenance/
├── app/app.py                      # Streamlit dashboard
├── data/README.md                  # dataset download instructions
├── notebooks/predictive_maintenance.ipynb
├── src/
│   ├── data_preprocessing.py       # load, clean, feature engineering
│   ├── train.py                    # train + evaluate + save best model
│   └── predict.py                  # CLI / functions for inference
├── models/                         # saved model + metrics (git-ignored)
├── requirements.txt
├── .env.example
└── .gitignore
```

## Quick start

```bash
git clone https://github.com/<your-username>/predictive-maintenance.git
cd predictive-maintenance

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env               # optional, defaults work
```

1. **Get the data** → save the CSV as `data/ai4i2020.csv` (see [data/README.md](data/README.md)).
2. **Train:**
   ```bash
   python -m src.train
   ```
3. **Predict (CLI):**
   ```bash
   python -m src.predict --type M --air-temp 300 --process-temp 310 --rpm 1500 --torque 40 --tool-wear 100
   python -m src.predict --csv data/ai4i2020.csv --out predictions.csv   # batch
   ```
4. **Run the web app:**
   ```bash
   streamlit run app/app.py
   ```

## Approach

- **Leakage prevention:** `TWF, HDF, PWF, OSF, RNF` (individual failure modes) are dropped because
  `Machine failure` is derived from them. `UDI` and `Product ID` are identifiers and are dropped too.
- **Feature engineering** (based on the failure mechanisms described in the dataset):
  - `temp_diff = process_temp − air_temp` (heat dissipation failure)
  - `power_w = torque × rpm × 2π/60` (power failure)
  - `wear_torque = tool_wear × torque` (overstrain failure)
- **Class imbalance:** stratified split, class-weighted models, and model selection by **PR-AUC**
  (accuracy is misleading at ~96.6% / 3.4%).
- **Models compared:** Logistic Regression, Random Forest, Gradient Boosting. The best by PR-AUC is saved to `models/model.joblib`.

## Configuration

Set in `.env` (see `.env.example`): `DATA_PATH`, `MODEL_PATH`, `METRICS_PATH`, `RANDOM_STATE`, `TEST_SIZE`.

## Possible improvements

- Threshold tuning to trade precision vs. recall based on downtime cost
- Hyperparameter search (`RandomizedSearchCV`) and SMOTE / cost-sensitive learning
- Multi-label prediction of the specific failure mode
- SHAP explanations, Docker image, CI with pytest

## Dataset citation

S. Matzka, "Explainable Artificial Intelligence for Predictive Maintenance Applications,"
*2020 Third International Conference on Artificial Intelligence for Industries (AI4I)*, 2020.

## License

MIT

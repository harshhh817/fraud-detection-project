# CLAUDE.md — Project context for Claude Code

This file tells Claude Code everything about this project so it can continue
seamlessly. (This is a student's final-year placement project — keep all code
beginner-friendly and heavily commented, and explain changes in plain language.)

## What this project is

An end-to-end **credit-card fraud detection** system:
- An **XGBoost** model trained on the Kaggle "Credit Card Fraud Detection"
  dataset (features V1–V28 + Amount, label `Class`: 0 normal / 1 fraud).
- Served as a **serverless API** (AWS Lambda + API Gateway) — the model runs
  **inside Lambda** so there is no always-on server cost. Intentionally FREE.
- Goal: impressive but explainable for placement interviews.

## Hard constraints (do not break these)

1. **Keep it free.** Never introduce a paid always-on resource (no SageMaker
   hosted endpoint, no EC2, no RDS). Lambda + API Gateway + DynamoDB free tiers
   only.
2. **Beginner-friendly.** The user is a student who must understand and defend
   every line. Comment generously; explain the "why" in plain English.
3. **Keep the two API entry points behaving identically** — the prediction
   logic lives ONCE as `predict()` in `app/lambda_function.py`, and
   `app/local_api.py` imports it from there. Never duplicate the logic.

## Current state (already built & tested)

- `data/generate_sample_data.py` — makes a fake dataset with the real schema so
  the project runs before the Kaggle download. Output: `data/creditcard_sample.csv`.
- `train/train_model.py` — trains XGBoost, handles imbalance via
  `scale_pos_weight`, prints precision/recall + ROC-AUC, **tunes the decision
  threshold (best F1, clamped 0.05–0.95)**, and runs an optional
  **scale_pos_weight vs SMOTE comparison** (needs `imbalanced-learn`). Saves
  `model/fraud_model.json`, `model/feature_columns.json`, `model/threshold.json`.
- `app/lambda_function.py` — AWS handler AND the single home of `predict()`.
  Reads the tuned threshold (falls back to 0.5 if missing). Has optional
  DynamoDB logging, off unless env var `FRAUD_LOG_TABLE` is set.
- `app/local_api.py` — Flask server for local testing (**port 5001**, because
  macOS AirPlay occupies 5000). Imports `predict()`; logs each prediction to
  `data/prediction_log.csv` for the dashboard.
- `app/*.json` — model + columns + threshold copies for Lambda packaging.
- `tests/test_predict.py` — 6 unit tests for `predict()` (`python -m pytest -v`).
- `dashboard/dashboard.py` — simple Streamlit monitoring dashboard; reads
  `data/prediction_log.csv` (`streamlit run dashboard/dashboard.py`).
- `Dockerfile` — Lambda container-image deploy path (xgboost is big).
- `web/index.html` — demo page with NORMAL/FRAUD example buttons (points at 5001).
- `deploy/AWS_DEPLOYMENT.md` — free, step-by-step AWS deploy + $1 budget alert;
  DynamoDB logging is now just "set the env var".
- `INTERVIEW_GUIDE.md` — viva Q&A.

Verified working end-to-end (July 2026): fraud row → flagged at ~1.0; normal
row → OK; all 6 tests pass; dashboard renders the prediction log.

**DEPLOYED & LIVE (July 2026)** — personal AWS account, region `eu-north-1`:
- Live API: `https://ry6rjihdqk.execute-api.eu-north-1.amazonaws.com/predict`
  (API Gateway HTTP API `fraud-api` → Lambda `fraud-detection`, x86_64,
  512 MB / 30 s, CORS `*` for POST/OPTIONS). Real metrics: recall 0.86.
- DynamoDB logging is ON: table `fraud-predictions` (partition key `id`),
  Lambda env var `FRAUD_LOG_TABLE=fraud-predictions`, role has DynamoDB access.
- Lambda zip lessons (see deploy guide): install Linux wheels
  (`--platform manylinux2014_x86_64`), model loads via native `xgboost.Booster`
  (NOT the sklearn wrapper) to avoid shipping scikit-learn, trim tests/.pyi to
  stay under the 50 MB console-upload limit.
- Demo the web page over http (`cd web && python3 -m http.server 8000` →
  `http://localhost:8000`); opening the file directly gives origin `null`,
  which CORS rejects.

**Gotcha:** `scikit-learn` is pinned to 1.5.2 in requirements.txt — 1.6+
removed an internal API that breaks `xgboost`'s `save_model()`.

## How to run

```bash
pip install -r requirements.txt
python data/generate_sample_data.py     # skip if you have real creditcard.csv
python train/train_model.py             # trains + tunes threshold + saves model
cp model/fraud_model.json model/feature_columns.json model/threshold.json app/
python -m pytest -v                     # 6 tests should pass
python app/local_api.py                 # starts API at http://127.0.0.1:5001
# then open web/index.html in a browser
streamlit run dashboard/dashboard.py    # optional: the monitoring dashboard
```

## Suggested next steps (ideas, not required)

- Add API-key auth via API Gateway.
- Write a short project report / slide deck from README + INTERVIEW_GUIDE.
- Retrain with the real Kaggle `creditcard.csv` before final submission.

## Style notes for Claude Code

- Prefer small, well-commented changes. After any change, tell the user in plain
  language what changed and why.
- Keep dependencies minimal (see requirements.txt).
- Don't add cloud resources that could incur cost without flagging it clearly.

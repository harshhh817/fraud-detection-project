# 💳 Credit Card Fraud Detection — End‑to‑End ML Project

A final‑year project that trains a machine‑learning model to detect fraudulent
credit‑card transactions and serves it as a **live prediction API**. Built to be
**100% free** to run and deploy, and written so a student can understand and
explain every part.

> **One‑line pitch (for your resume/interview):**
> "I built and deployed an XGBoost model that detects fraudulent card
> transactions in real time through a serverless API on AWS (Lambda + API
> Gateway), handling severe class imbalance and logging predictions for a
> monitoring dashboard."

---

## 1. What problem does this solve?

Banks process millions of card transactions. A tiny fraction are **fraud**
(stolen cards, etc.). Humans can't check every transaction, so we train a model
to look at a transaction and instantly answer: **"Is this fraud — yes or no?"**

The hard part isn't the model — it's that fraud is **rare** (under 1% of
transactions). This is called an **imbalanced** problem, and handling it
correctly is exactly what makes this project impressive.

---

## 2. How the whole thing works (the big picture)

```
   ┌─────────────┐     ┌──────────────┐     ┌────────────────────┐
   │  Transaction │ ──▶ │  API Gateway │ ──▶ │  Lambda function   │
   │  (from a UI) │     │  (public URL)│     │  loads the model   │
   └─────────────┘     └──────────────┘     │  and predicts      │
                                            └─────────┬──────────┘
                                                      │
                          ┌───────────────────────────┼───────────────┐
                          ▼                                            ▼
                   ┌──────────────┐                          ┌──────────────────┐
                   │ Answer sent  │                          │ Prediction logged │
                   │ back: FRAUD  │                          │ to DynamoDB, shown│
                   │ or OK        │                          │ on a dashboard    │
                   └──────────────┘                          └──────────────────┘
```

**In words:** You train the model once on your laptop (or free Google Colab).
You save it as a small file. You put that file inside an AWS Lambda function.
When a transaction arrives at your public API URL, Lambda loads the model,
predicts fraud/not‑fraud in milliseconds, sends the answer back, and logs it.

The clever, money‑saving trick: **the model runs INSIDE Lambda**, so you do NOT
pay for an always‑on server. That is why this project is free. (See
`deploy/AWS_DEPLOYMENT.md`.)

---

## 3. What's in this folder

```
fraud-detection-project/
├── README.md                     ← you are here (start here)
├── requirements.txt              ← the Python libraries to install
├── data/
│   ├── generate_sample_data.py   ← makes fake data so the project runs instantly
│   └── creditcard_sample.csv     ← the fake data (created when you run it)
├── train/
│   └── train_model.py            ← trains the model, tunes the threshold, SMOTE compare
├── model/
│   ├── fraud_model.json          ← the trained model (created by training)
│   ├── feature_columns.json      ← the exact input order the model expects
│   └── threshold.json            ← the tuned fraud cut-off (created by training)
├── app/
│   ├── lambda_function.py        ← the code that runs on AWS (the shared "brain")
│   └── local_api.py              ← run the SAME logic on your laptop to test
├── tests/
│   └── test_predict.py           ← unit tests for the prediction logic
├── dashboard/
│   └── dashboard.py              ← Streamlit "fraud monitoring" dashboard
├── web/
│   └── index.html                ← a simple web page to click and test
├── deploy/
│   └── AWS_DEPLOYMENT.md          ← free, step‑by‑step AWS deployment
├── Dockerfile                    ← optional: package Lambda as a container image
└── INTERVIEW_GUIDE.md            ← answers to the questions you'll be asked
```

---

## 4. Run it on your laptop in 4 steps (no AWS needed)

You need **Python 3.9+** installed. Then, from inside this folder:

**Step 1 — install the libraries**
```
pip install -r requirements.txt
```

**Step 2 — create the sample data** (skip if you have the real Kaggle file)
```
python data/generate_sample_data.py
```

**Step 3 — train the model** (this also prints your scores)
```
python train/train_model.py
```
You'll see a report with **precision** and **recall** (explained below), and a
`fraud_model.json` file will appear in the `model/` folder.

**Step 4 — start the API and test it**
```
python app/local_api.py
```
Now open `web/index.html` in your browser, click **"Load FRAUD example"**, then
**"Check transaction"**. You should see it flagged as fraud. Try the NORMAL
example too.

That's the entire project working locally. 🎉

---

## 5. Using the REAL dataset (do this for your final submission)

The sample data is fake — good enough to test the code, but for your real
project use the genuine dataset so your results are meaningful:

1. Go to Kaggle → search **"Credit Card Fraud Detection"** (by ULB / MLG).
2. Download `creditcard.csv` (it's free; you just need a Kaggle account).
3. Put `creditcard.csv` into the `data/` folder.
4. Run `python train/train_model.py` again — it **automatically** uses the real
   file when it's there.

---

## 6. The two words that win interviews: precision & recall

Because fraud is rare, **accuracy is a trap**. A model that says "not fraud"
for everything is 99.8% accurate and catches ZERO fraud. So we use:

- **Recall** = of all the REAL frauds, how many did we catch? (Catch the crooks.)
- **Precision** = of everything we FLAGGED as fraud, how many really were fraud?
  (Don't annoy honest customers with false alarms.)

There's a **trade‑off**: catch more fraud (high recall) and you get more false
alarms (lower precision). Being able to say this out loud is what separates you
from someone who just prints "accuracy = 0.99".

**How we handle the imbalance in code:** in `train/train_model.py` we set
`scale_pos_weight`, which tells XGBoost to pay far more attention to the rare
fraud cases so it actually learns to catch them.

---

## 7. The extra features (great to demo in interviews)

All of these are **free** and optional. Install the extras first:
`pip install pytest imbalanced-learn streamlit boto3`.

- **Unit tests** — prove the prediction code works:
  ```
  python -m pytest -v
  ```
  (Train the model and copy the files into `app/` first — see the note below.)

- **Tuned decision threshold** — `train/train_model.py` no longer blindly uses
  0.5. It scans every cut-off, picks the one with the best F1 score, and saves
  it to `model/threshold.json`. The API reads this automatically (and safely
  falls back to 0.5 if the file is missing).

- **scale_pos_weight vs SMOTE** — training prints a fair comparison (PR-AUC) of
  the two classic ways to handle imbalance, so you can explain the trade-off.

- **Monitoring dashboard** — every local prediction is logged to
  `data/prediction_log.csv`; on AWS you can log to DynamoDB instead (set the
  `FRAUD_LOG_TABLE` env var). See it live:
  ```
  streamlit run dashboard/dashboard.py
  ```

- **Docker (container deploy)** — a `Dockerfile` is included for the Lambda
  container-image path (handy because XGBoost is large). See
  `deploy/AWS_DEPLOYMENT.md`.

> **After training, copy the model files into `app/`** so the API/tests/Lambda
> can find them:
> ```
> cp model/fraud_model.json model/feature_columns.json model/threshold.json app/
> ```

---

## 8. Deploying to AWS for free

See **`deploy/AWS_DEPLOYMENT.md`** for the full click‑by‑click guide, plus the
**$1 budget alert** you should set so you can never be surprised by a bill.

## 9. Preparing for the viva / interview

See **`INTERVIEW_GUIDE.md`** for the exact questions you'll be asked and clear,
correct answers you can say in your own words.

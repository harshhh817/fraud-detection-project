# 📚 Complete Study Guide — Credit Card Fraud Detection Project

**Read this to understand the WHOLE project from zero.** It assumes you know
nothing. It explains every component, every important line of code, how each
piece fits into the workflow, and the concepts behind it — in plain English.

> How to use this guide: read Parts 1–3 first (the big picture and the key
> ideas). Then go file-by-file in Part 5 with the actual code open beside you.
> Part 9 is a quick revision sheet for the night before your viva.

---

## Table of contents

1. [The big picture — what and why](#part-1)
2. [The two workflows: "train once" and "predict many"](#part-2)
3. [Key concepts you MUST understand (plain English)](#part-3)
4. [The project folder — every file and its job](#part-4)
5. [Deep dive: every file explained line-by-line](#part-5)
6. [The AWS deployment — each service explained](#part-6)
7. [The full request lifecycle (everything tied together)](#part-7)
8. [Which code contributes to which feature](#part-8)
9. [Viva quick-revision sheet](#part-9)
10. [Glossary of every term](#part-10)

---

<a name="part-1"></a>
## Part 1 — The big picture: what and why

### The problem
Banks process **millions** of card transactions. A tiny fraction (well under 1%)
are **fraud** — someone using a stolen card. Humans cannot check every
transaction by hand, so we train a **computer model** to look at a transaction
and instantly answer one question: **"Is this fraud — yes or no?"**

### What we built
An **end-to-end machine-learning system**. "End-to-end" means it covers the
whole journey: from raw data → to a trained model → to a live API on the
internet → to a web page a person can click → plus monitoring. The pieces:

- A **model** (XGBoost) trained on real transaction data.
- A **serverless API** (AWS Lambda + API Gateway) that runs the model and is
  reachable at a public web address.
- A **web page** to demo it, and a **dashboard** to monitor it.

### The one design decision that defines the project: keep it FREE
The model runs **inside** AWS Lambda. Lambda only runs (and only costs money)
when a request actually arrives — the rest of the time it costs **nothing**.
We deliberately avoid any "always-on" server (which would bill you 24/7). This
is both free *and* a strong talking point: "I made a cost-conscious architecture
choice." See [Part 6](#part-6).

---

<a name="part-2"></a>
## Part 2 — The two workflows

The single most important thing to understand: this project has **two separate
jobs**, and different files do each one.

### Workflow A — "Train the model" (happens ONCE, on your laptop)
You run this occasionally — when you first build the project, or when you get new
data. It reads past transactions, learns the fraud patterns, and saves the
learned model into a small file.

```
creditcard.csv  ──►  train/train_model.py  ──►  model/fraud_model.json
 (past data)          (learns the patterns)      (the saved "brain")
```

### Workflow B — "Use the model" (happens EVERY time someone checks a transaction)
This is the live part. A transaction comes in, the saved model scores it, and an
answer goes back — in milliseconds.

```
web page  ──►  API Gateway  ──►  Lambda (loads fraud_model.json)  ──►  FRAUD / OK
                                        │
                                        └──►  DynamoDB (logs it)  ──►  dashboard
```

**The bridge between the two workflows is a single file: `fraud_model.json`.**
Training *writes* it; the API *reads* it. They never run at the same time and
never talk directly — the file is the only thing they share.

### Which file belongs to which workflow

| File | Workflow | Role |
|------|----------|------|
| `data/generate_sample_data.py` | A (setup) | Makes fake data so the project runs before you download the real data |
| `train/train_model.py` | A | Trains the model, saves the files |
| `model/fraud_model.json` etc. | bridge | The saved model (output of A, input to B) |
| `app/lambda_function.py` | B | The prediction "brain" + the AWS entry point |
| `app/local_api.py` | B | Runs the same brain on your laptop for testing |
| `web/index.html` | B | The web page a person clicks |
| `dashboard/dashboard.py` | B (monitoring) | Shows logged predictions over time |
| `tests/test_predict.py` | quality | Automated checks that prediction works |

---

<a name="part-3"></a>
## Part 3 — Key concepts you MUST understand

If you understand these eight ideas, you understand the project. Everything else
is detail.

### 1. Features and labels (the `X` and `y`)
- A **feature** is an input — a fact about the transaction (its amount, its V1
  value, etc.). All the features together are called **`X`**.
- The **label** (also "target") is the answer we want to predict — here the
  `Class` column: `0` = normal, `1` = fraud. The label is called **`y`**.
- Training = showing the model many rows of `X` **with** their correct `y`, so
  it learns the relationship "these kinds of X usually mean fraud."

### 2. The `V1–V28` features (and PCA)
Each transaction is described by **30 numbers**: `Time`, `V1…V28`, and `Amount`.
The `V` numbers are **28 different aspects of the same single transaction** (not
28 transactions!) — like how one person needs several numbers to describe (height,
weight, age...). The original details (merchant, location, etc.) were real bank
data, so before release they were scrambled with a maths technique called **PCA**
(Principal Component Analysis) that **hides the meaning but keeps the patterns**.
So `V1–V28` are information-rich but deliberately unreadable, for privacy.

### 3. Class imbalance (the heart of the project)
Fraud is **rare** — under 1% of rows. This breaks the naive approach: a lazy
model that always says "normal" would be **99.8% accurate** and catch **zero**
fraud. So **accuracy is a trap** here. Two things fix this:
- We measure the *right* scores (precision & recall, below), not accuracy.
- We tell the model to care more about the rare class (`scale_pos_weight`).

### 4. Precision, recall, and F1 (the scores that matter)
- **Recall** = of all the REAL frauds, how many did we catch? *(Catch the crooks.)*
- **Precision** = of everything we FLAGGED as fraud, how many were truly fraud?
  *(Don't annoy honest customers with false alarms.)*
- **F1 score** = a single number that balances precision and recall (their
  "harmonic mean"). Higher is better. We use F1 to pick our threshold.
- There is a **trade-off**: catch more fraud (higher recall) → more false alarms
  (lower precision). The right balance depends on business cost.

### 5. The decision threshold
The model outputs a **probability** of fraud (a number from 0 to 1). To turn that
into a yes/no we need a **cut-off**: "if probability ≥ threshold, call it fraud."
The default is `0.5`, but that's just a convention. We **tune** the threshold to
whatever gives the best F1 on our data (ours came out around `0.86`).

### 6. XGBoost (the model type)
**XGBoost** = "eXtreme Gradient Boosting." It builds hundreds of small **decision
trees**, each one correcting the mistakes of the previous ones ("boosting"). It's
the go-to choice for **tabular data** (rows and columns, like a spreadsheet) — it
usually beats neural networks on this kind of data, trains fast, and is small
enough to deploy. We do **not** use deep learning because that shines on
images/text, not spreadsheet-style data.

### 7. Serverless & AWS Lambda
A traditional server runs 24/7 and bills you 24/7, even when idle. **Serverless**
(AWS **Lambda**) flips this: your code sits dormant and only **runs on demand**
when a request arrives — you pay only for those milliseconds. Lambda gives 1
million free requests/month, forever. That's why this project is free.
- **Cold start**: the *first* request after idle has to load the code + model, so
  it's a little slow (~1 sec). After that the function stays "warm" and is
  instant. (Great viva term.)

### 8. The API, API Gateway, and CORS
- An **API** is a way for one program to ask another program for something over
  the internet, using a URL. Our API takes a transaction and returns a verdict.
- **API Gateway** is the AWS service that gives your Lambda a **public URL** and
  routes incoming requests to it.
- **CORS** (Cross-Origin Resource Sharing) is a browser security rule. A web page
  can only call an API on a *different* domain if that API explicitly says "I
  allow this." We configured CORS so browsers are permitted to call our API.

---

<a name="part-4"></a>
## Part 4 — The project folder: every file and its job

```
fraud-detection-project/
│
├── data/
│   ├── generate_sample_data.py   Makes fake data (so it runs before Kaggle)
│   ├── creditcard_sample.csv     The fake data (created by the script above)
│   └── creditcard.csv            The REAL Kaggle data (you downloaded this)
│
├── train/
│   └── train_model.py            Trains the model + tunes threshold + saves files
│
├── model/                        OUTPUT of training (the "bridge" files)
│   ├── fraud_model.json          The trained model itself
│   ├── feature_columns.json      The exact order of the 30 input columns
│   └── threshold.json            The tuned fraud cut-off (~0.86)
│
├── app/
│   ├── lambda_function.py        THE BRAIN: predict() + AWS handler + logging
│   ├── local_api.py              Runs the brain on your laptop (Flask server)
│   ├── fraud_model.json          Copy of the model (Lambda needs it here)
│   ├── feature_columns.json      Copy (same reason)
│   └── threshold.json            Copy (same reason)
│
├── tests/
│   └── test_predict.py           Automated tests for predict()
│
├── dashboard/
│   └── dashboard.py              Streamlit monitoring dashboard
│
├── web/
│   └── index.html                The clickable demo web page
│
├── deploy/
│   └── AWS_DEPLOYMENT.md          Step-by-step AWS deployment guide
│
├── Dockerfile                    Optional: package Lambda as a container image
├── requirements.txt              The Python libraries to install
├── README.md                     Project overview
├── INTERVIEW_GUIDE.md            Viva questions & answers
├── STUDY_GUIDE.md                ← this file
└── CLAUDE.md                     Project context / notes
```

**Why are the model files in TWO places (`model/` and `app/`)?** Training saves
them to `model/`. But when we zip up the app for AWS Lambda, Lambda needs the
model sitting *next to* `lambda_function.py` (in `app/`). So after training we
**copy** the three files from `model/` into `app/`. That's the one manual step:
```
cp model/fraud_model.json model/feature_columns.json model/threshold.json app/
```

---

<a name="part-5"></a>
## Part 5 — Deep dive: every file explained

For each file: what it's for, then the important lines explained.

---

### 5.1 — `data/generate_sample_data.py` (make fake data)

**Purpose:** the real Kaggle data needs an account to download. So the project
can run *immediately*, this script invents fake data with the **exact same
columns** (`Time, V1..V28, Amount, Class`). You throw it away once you have the
real `creditcard.csv`.

```python
np.random.seed(42)          # makes the "random" data the same every run (repeatable)
N_NORMAL = 9800             # 9800 normal rows
N_FRAUD = 200               # only 200 fraud rows -> deliberately imbalanced (~2%)
```

```python
shift = 2.0 if is_fraud else 0.0
v = np.random.normal(loc=shift, scale=1.0, size=(n, 28))
```
This makes the 28 `V` columns as random numbers. The trick: **fraud rows are
shifted up by 2**, so there's a fake "pattern" the model can learn. (Real data
has genuine patterns; this fake shift just lets the code run end-to-end.)

```python
df = pd.concat([normal, fraud]).sample(frac=1, ...)   # stack + shuffle
df.to_csv(out_path, index=False)                       # save as CSV
```
Combines normal + fraud rows, shuffles them, and writes `creditcard_sample.csv`.

> **Interview note:** the *fake* fraud pattern (all V's shifted +2) is NOT what
> real fraud looks like. That's why, after switching to real data, the old
> "all V's = 2.2" demo example stopped being flagged — the real model learned
> real patterns. We updated the demo to use genuine rows from the dataset.

---

### 5.2 — `train/train_model.py` (the training workflow) ⭐

This is the most important file to understand for the ML side. It runs in 6
steps. Open the file and follow along.

**Imports (lines 17–26):** `pandas` (tables), `numpy` (numbers),
`scikit-learn` (the split + the scoring metrics), and `xgboost` (the model).

**Step 1 — Load the data (lines 38–55):**
```python
if os.path.exists(REAL_DATA):     # if creditcard.csv is present, use it
    data_path = REAL_DATA
else:
    data_path = SAMPLE_DATA       # otherwise fall back to the fake data
df = pd.read_csv(data_path)
X = df.drop(columns=["Class"])    # X = all inputs (every column except the answer)
y = df["Class"]                   # y = the answer (0 normal / 1 fraud)
```
This is the **features/labels** split from Part 3. `X` is the 30 input columns,
`y` is the label.

**Step 2 — Train/test split (lines 58–64):**
```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y)
```
We hide **20%** of the data as a **test set** the model never sees while
learning. Then we score the model on that unseen 20% — that's the only honest way
to know if it really learned (vs. just memorised). `stratify=y` keeps the same
tiny fraud ratio in both halves. `random_state=42` makes the split repeatable.

**Step 3 — Handle the imbalance (lines 67–75):**
```python
n_normal = (y_train == 0).sum()
n_fraud = (y_train == 1).sum()
scale = n_normal / max(n_fraud, 1)     # e.g. 227000 / 400 ≈ 577
```
`scale_pos_weight` will tell XGBoost to weight each rare fraud case ~577× more
heavily, so it can't get away with ignoring them. (`max(n_fraud, 1)` just avoids
dividing by zero.)

**Step 4 — Train the model (lines 78–89):**
```python
model = xgb.XGBClassifier(
    n_estimators=200,        # build 200 trees
    max_depth=4,             # each tree max 4 levels deep (small = avoids overfitting)
    learning_rate=0.1,       # how big each correction step is
    scale_pos_weight=scale,  # ← the imbalance fix from Step 3
    eval_metric="aucpr",     # judge itself using a metric good for imbalance
    random_state=42,
    n_jobs=-1)               # use all CPU cores
model.fit(X_train, y_train)  # ← THIS LINE is where the actual learning happens
```
`model.fit(...)` is the moment the model looks at the training rows and learns.

**Step 5 — Evaluate properly (lines 92–105):**
```python
y_pred = model.predict(X_test)                 # yes/no guesses on unseen data
y_proba = model.predict_proba(X_test)[:, 1]    # the raw fraud probabilities
print(confusion_matrix(y_test, y_pred))        # table of right/wrong counts
print(classification_report(y_test, y_pred))   # precision, recall, F1 per class
print(roc_auc_score(y_test, y_proba))          # overall separation score (0.5=random,1=perfect)
```
This prints the **precision/recall** we care about (Part 3), computed on the
held-out test set.

- The **confusion matrix** is a 2×2 table: `[[true-normal, false-alarm],
  [missed-fraud, caught-fraud]]`. Our real-data run:
  `[[56817, 47], [14, 84]]` → caught 84 frauds, missed 14, 47 false alarms.

**Step 5b — Threshold tuning (lines 108–136):**
```python
precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)
best_threshold, best_f1 = 0.5, -1.0
for p, r, t in zip(precisions[:-1], recalls[:-1], thresholds):
    f1 = 2 * p * r / (p + r)          # F1 at this candidate cut-off
    if f1 > best_f1:
        best_f1, best_threshold = f1, float(t)   # keep the best one
best_threshold = min(max(best_threshold, 0.05), 0.95)   # keep it sensible
```
Instead of blindly using 0.5, we try **every** possible cut-off and keep the one
with the best F1. The clamp to `0.05–0.95` avoids a silly extreme value on easy
data. Result on real data: threshold `0.86`, which lifted F1 from `0.73` → `0.85`.

**Step 5c — scale_pos_weight vs SMOTE (lines 139–171):**
An optional experiment: **SMOTE** is another way to fix imbalance (it invents
synthetic fraud examples to balance the classes). We train a second model with
SMOTE and compare the two using **PR-AUC** (a fair single score for imbalance).
Purpose: to *show you understand the trade-off*. We keep `scale_pos_weight`
because it's simpler and needs no invented data. Wrapped in `try/except ImportError`
so it's skipped cleanly if the `imbalanced-learn` library isn't installed.

**Step 6 — Save everything (lines 174–196):**
```python
model.save_model(model_path)                          # -> model/fraud_model.json
json.dump(list(X.columns), open(columns_path, "w"))   # -> feature_columns.json (the order)
json.dump({"threshold": best_threshold}, open(threshold_path, "w"))  # -> threshold.json
```
Three output files — the **bridge** to the prediction workflow. `feature_columns.json`
matters because the model expects features in the **exact same order** it trained
on; the API uses this file to line the inputs up correctly.

---

### 5.3 — The model artifacts (the "bridge" files)

These are *created by training*, *read by the API*. You don't edit them by hand.

- **`fraud_model.json`** — the trained model: all 200 trees and their rules, in
  XGBoost's own format. This single file *is* the learned brain.
- **`feature_columns.json`** — a list like `["Time","V1",...,"V28","Amount"]`.
  The order the model expects its inputs in.
- **`threshold.json`** — one number, e.g. `{"threshold": 0.8619}`. The tuned
  fraud cut-off.

---

### 5.4 — `app/lambda_function.py` (THE BRAIN + AWS entry point) ⭐

The most important file for the deployment side. It does three jobs: (1) it holds
the one shared `predict()` function, (2) it optionally logs to DynamoDB, and (3)
it provides `lambda_handler`, the function AWS calls.

**Load the model ONCE (lines 31–60).** This code runs a single time when the
Lambda "wakes up" (cold start), not on every request — that's what makes later
requests fast.
```python
_booster = xgb.Booster()          # xgboost's NATIVE model object...
_booster.load_model(MODEL_PATH)   # ...load the trained model file into memory
FEATURE_COLUMNS = json.load(open(COLUMNS_PATH))   # the column order
THRESHOLD = _load_threshold()     # the tuned cut-off (or 0.5 if file missing)
```
> **Why `Booster` and not `XGBClassifier`?** The training file used the friendly
> `XGBClassifier` wrapper, but that wrapper needs **scikit-learn** installed at
> prediction time — a big library that would blow our Lambda size limit. The
> native `Booster` gives *identical* predictions with **no scikit-learn needed**,
> keeping the deployment small. This was a real problem we hit and fixed. (Great
> viva story.)

**The shared `predict()` function (lines 63–85)** — the actual decision logic:
```python
def predict(transaction: dict):
    row = [float(transaction.get(col, 0)) for col in FEATURE_COLUMNS]
    #   ^ build the input in the trained column order; any missing feature -> 0
    dmatrix = xgb.DMatrix(np.array([row]))         # xgboost's input format
    proba = float(_booster.predict(dmatrix, validate_features=False)[0])
    #   ^ the model's fraud probability, 0..1
    is_fraud = proba >= THRESHOLD                  # apply the tuned cut-off
    return {
        "is_fraud": bool(is_fraud),
        "fraud_probability": round(proba, 4),
        "decision": "FRAUD - review this transaction" if is_fraud else "OK - looks normal",
    }
```
Line by line:
- `transaction.get(col, 0)` — read each feature by name; if it's missing, use `0`.
  This is why the API never crashes on incomplete input.
- The list comprehension puts values in the **exact trained order** (`FEATURE_COLUMNS`).
- `DMatrix` is xgboost's optimised input container.
- `_booster.predict(...)[0]` — because the model's objective is `binary:logistic`,
  this returns the fraud probability directly.
- `proba >= THRESHOLD` — turns the probability into a yes/no using our tuned cut-off.
- Returns a dictionary with three fields (the JSON the caller gets back).

**Optional DynamoDB logging (lines 88–112):**
```python
def log_prediction_dynamodb(transaction, result):
    table_name = os.environ.get("FRAUD_LOG_TABLE")   # read the env var
    if not table_name:
        return                                        # not set -> logging OFF
    try:
        import boto3
        table = boto3.resource("dynamodb").Table(table_name)
        table.put_item(Item={...})                    # write one row
    except Exception as e:
        print(f"[warn] could not log: {e}")           # never crash on log failure
```
- It only does anything if the environment variable `FRAUD_LOG_TABLE` is set (we
  set it to `fraud-predictions` on AWS). Locally it's unset, so this is a no-op.
- The whole thing is inside `try/except` so **logging can never break a
  prediction** — monitoring must never take down the thing it monitors.

**The AWS entry point `lambda_handler` (lines 115–144):**
```python
def lambda_handler(event, context):
    try:
        if "body" in event and event["body"] is not None:
            body = json.loads(event["body"])   # real API request: JSON is a string
        else:
            body = event                        # console test: already a dict
        result = predict(body)                  # ← run the brain
        log_prediction_dynamodb(body, result)   # ← optional logging
        return {
            "statusCode": 200,
            "headers": {"Access-Control-Allow-Origin": "*", ...},  # CORS + JSON
            "body": json.dumps(result),
        }
    except Exception as e:
        return {"statusCode": 400, ..., "body": json.dumps({"error": str(e)})}
```
- `event` is what AWS hands in. From API Gateway, the transaction JSON arrives as
  a **string** inside `event["body"]`, so we `json.loads` it. From the Lambda
  test console, it's already a dict — we handle both.
- On success: HTTP `200` with the result. On any error: HTTP `400` with the
  message (so the caller gets a clear error, not a crash).
- `Access-Control-Allow-Origin: *` is the CORS header.

---

### 5.5 — `app/local_api.py` (run the brain on your laptop)

**Purpose:** test the exact same logic without AWS. It's a tiny **Flask** web
server. The key idea: it does **not** re-implement `predict()` — it **imports**
it from `lambda_function.py`, so local and AWS behaviour are guaranteed identical.

```python
sys.path.insert(0, HERE)               # let Python find lambda_function.py
from lambda_function import predict     # ← the SAME brain, imported
```

```python
@app.route("/predict", methods=["POST"])
def predict_route():
    body = request.get_json(force=True)   # read the JSON the browser sent
    result = predict(body)                 # run the shared brain
    log_prediction_local(body, result)     # log to a local CSV (for the dashboard)
    return jsonify(result)                  # send the answer back as JSON
```
- `log_prediction_local` writes each prediction to `data/prediction_log.csv` —
  the local equivalent of the DynamoDB logging, so the dashboard has data to show
  even without AWS.
- Runs on **port 5001** (not 5000, because macOS uses 5000 for AirPlay).

---

### 5.6 — `tests/test_predict.py` (automated quality checks)

**Purpose:** prove the prediction logic works, automatically. Run with
`python -m pytest -v`. Answers the viva question "how do you know your code
works?"

Each `test_...` function checks one thing:
```python
def test_normal_transaction_is_not_flagged():
    assert predict(NORMAL)["is_fraud"] is False   # a real normal row -> not fraud

def test_fraud_transaction_is_flagged():
    result = predict(FRAUD)
    assert result["is_fraud"] is True             # a real fraud row -> fraud
    assert result["fraud_probability"] > 0.5

def test_missing_features_default_to_zero():
    assert "is_fraud" in predict({"Amount": 100})  # incomplete input -> no crash
```
`assert X` means "if X is not true, fail this test." The `NORMAL` and `FRAUD`
constants are **real rows from the Kaggle dataset**, so the tests check the model
against genuine behaviour. All 6 pass.

---

### 5.7 — `web/index.html` (the demo web page)

**Purpose:** a simple page with a text box and buttons so a human can try the
model. Pure HTML + JavaScript, no framework.

The important JavaScript:
```javascript
const res = await fetch(url, {                 // send the transaction to the API
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload)               // the transaction as JSON text
});
const data = await res.json();                  // read the API's answer
if (data.is_fraud) { box.className = 'fraud'; ... }   // red banner
else { box.className = 'ok'; ... }                    // green banner
```
- `fetch(url, ...)` is the browser calling your API over the internet.
- `url` defaults to your live AWS endpoint (baked into the page).
- The `Load FRAUD example` / `Load NORMAL example` buttons paste real dataset
  rows into the box so you can demo instantly.

> **Must serve over http, not file://.** Opening the file directly gives the
> browser the origin `null`, which CORS rejects. Run
> `cd web && python3 -m http.server 8000` and open `http://localhost:8000`.

---

### 5.8 — `dashboard/dashboard.py` (monitoring)

**Purpose:** the "monitoring story." A one-page **Streamlit** app that reads the
prediction log and shows totals, a fraud rate, a chart over time, and a table.
Run with `streamlit run dashboard/dashboard.py`.

```python
df = pd.read_csv(LOG_FILE)                       # read the logged predictions
df["time"] = pd.to_datetime(df["timestamp"], unit="s")  # make timestamps readable

col1.metric("Transactions checked", total)       # headline numbers ("KPIs")
col2.metric("Flagged as fraud", frauds)
col3.metric("Fraud rate", f"{frauds/total*100:.1f}%")

per_hour = frauds_only.set_index("time").resample("1h").size()
st.bar_chart(per_hour)                            # fraud flags per hour
st.dataframe(df.sort_values("time", ascending=False).head(50))  # recent rows
```
- Reads `data/prediction_log.csv` (written by the local API). On AWS you'd point
  it at the DynamoDB table instead.
- `st.metric`, `st.bar_chart`, `st.dataframe` are Streamlit's ready-made widgets —
  a few lines gives a real dashboard.

---

### 5.9 — `Dockerfile` (optional container deployment)

**Purpose:** an *alternative* way to package the app for Lambda, as a **container
image** instead of a zip. Useful because xgboost is large (containers allow up to
10 GB vs the zip's 250 MB). We deployed with the zip, so this is a backup path.
```dockerfile
FROM public.ecr.aws/lambda/python:3.12   # start from AWS's official Lambda image
RUN pip install --no-cache-dir xgboost numpy   # install the runtime libraries
COPY app/lambda_function.py ${LAMBDA_TASK_ROOT}/   # copy handler + model in
...
CMD [ "lambda_function.lambda_handler" ]   # tell Lambda which function to run
```

---

### 5.10 — `requirements.txt` (the shopping list of libraries)

Lists the Python libraries to install with `pip install -r requirements.txt`.
Split into **core** (needed to train + run) and **optional** (tests, SMOTE,
dashboard, AWS).
- `numpy, pandas` — number and table handling.
- `scikit-learn==1.5.2` — the split + metrics. **Pinned** to 1.5.2 because newer
  versions broke xgboost's model saving (a real bug we hit).
- `xgboost` — the model.
- `flask` — the local web server.
- `pytest, imbalanced-learn, streamlit, boto3` — optional extras.

---

<a name="part-6"></a>
## Part 6 — The AWS deployment, explained

Four AWS services work together. Here's what each does, in plain English.

| Service | Plain-English job | In our project |
|---------|-------------------|----------------|
| **Lambda** | Runs your code on demand, no server to manage | Holds the model + `predict()`; runs on each request |
| **API Gateway** | Gives your Lambda a public URL and routes requests | Turns a web address into a Lambda call (`POST /predict`) |
| **DynamoDB** | A simple, fast, serverless database | Stores each prediction for the dashboard |
| **IAM** | Permissions — who is allowed to do what | Lets the Lambda write to DynamoDB (via a "role") |

### What we actually did (and the four real problems we solved)
1. **Built a deployment zip** of the code + model + libraries.
   - *Problem:* a plain `pip install` on a Mac bundles **Mac** binaries, which
     crash on Lambda's **Linux**. *Fix:* install Linux wheels with
     `--platform manylinux2014_x86_64`.
   - *Problem:* the zip was over the **50 MB** console limit. *Fix:* trim test
     files and type-stubs.
2. **Created the Lambda function** `fraud-detection` (Python 3.12, x86_64, 512 MB,
   30 s timeout), uploaded the zip, set the handler to
   `lambda_function.lambda_handler`.
   - *Problem:* it needed **scikit-learn**. *Fix:* switched to the native
     `Booster` (see 5.4) so no sklearn is needed.
3. **Added API Gateway** (an HTTP API with a `POST /predict` route) → got the
   public URL.
   - *Problem:* the browser was blocked by **CORS**. *Fix:* enabled CORS on the
     API (via the AWS CLI in CloudShell).
4. **Added DynamoDB logging**: created the `fraud-predictions` table, gave the
   Lambda permission (IAM role), and set the `FRAUD_LOG_TABLE` env var to switch
   logging on.

Our live endpoint:
`https://ry6rjihdqk.execute-api.eu-north-1.amazonaws.com/predict`

---

<a name="part-7"></a>
## Part 7 — The full request lifecycle (everything tied together)

What actually happens, start to finish, when someone checks a transaction on the
live system:

1. **You** open `http://localhost:8000` and click "Check transaction".
2. **`web/index.html`** runs `fetch(...)`, sending the transaction JSON to the
   API Gateway URL.
3. The browser first sends a **CORS preflight** (an `OPTIONS` request). API
   Gateway answers "yes, allowed" because we configured CORS.
4. The real **POST** arrives at **API Gateway**, which routes it to the
   **Lambda** function.
5. **Lambda** runs `lambda_handler` in `lambda_function.py`:
   - reads the JSON body,
   - calls `predict()`, which feeds the numbers to the **XGBoost model**
     (already loaded in memory) and gets a probability,
   - applies the **threshold** to decide fraud/not,
   - writes a row to **DynamoDB** (logging),
   - returns the result as JSON with a **CORS header**.
6. **API Gateway** passes the JSON back to the browser.
7. **`web/index.html`** reads the answer and shows a **red 🚨 FRAUD** or
   **green ✅ OK** banner with the probability.
8. Later, **`dashboard.py`** reads the logged rows and shows the trend.

All of steps 2–7 take a few hundred milliseconds (a bit longer on a cold start).

---

<a name="part-8"></a>
## Part 8 — Which code contributes to which feature

A quick map from "project feature" → "the code that makes it happen."

| Feature of the project | Where it lives |
|------------------------|----------------|
| Handling class imbalance | `train_model.py` Step 3 (`scale_pos_weight`) |
| Proper scoring (precision/recall) | `train_model.py` Step 5 |
| Threshold tuning | `train_model.py` Step 5b → `threshold.json` → used in `predict()` |
| SMOTE comparison | `train_model.py` Step 5c |
| The actual prediction | `predict()` in `lambda_function.py` |
| "Two entry points stay identical" | `local_api.py` imports `predict` from `lambda_function.py` |
| Runs on AWS | `lambda_handler` in `lambda_function.py` |
| Public URL | API Gateway (`POST /predict`) |
| Free / serverless | The whole Lambda-based architecture (no always-on server) |
| Monitoring / logging | `log_prediction_dynamodb()` + `dashboard.py` |
| Small deployment package | Native `Booster` (no sklearn) + Linux wheels + trimming |
| Browser access | CORS headers + API Gateway CORS config |
| Quality assurance | `tests/test_predict.py` |
| Runs before real data | `generate_sample_data.py` |

---

<a name="part-9"></a>
## Part 9 — Viva quick-revision sheet

**30-second pitch:** "I built and deployed an XGBoost model that detects
fraudulent card transactions in real time through a serverless API on AWS
(Lambda + API Gateway). I handled severe class imbalance, tuned the decision
threshold, log every prediction to DynamoDB for a monitoring dashboard, and
covered the prediction code with unit tests. It's fully serverless so there's no
idle cost."

**The 10 questions you'll likely get:**
1. *What does it do?* → Predicts fraud/not-fraud for a transaction, in real time,
   via a web API.
2. *Why is the data hard?* → Fraud is <1% (imbalanced); accuracy is misleading.
3. *How did you fix imbalance?* → `scale_pos_weight`; also compared SMOTE.
4. *Why precision/recall not accuracy?* → Accuracy hides failure on the rare
   class. Recall = frauds caught; precision = flags that were right.
5. *What's the threshold?* → The probability cut-off for "fraud"; I tuned it for
   best F1 (~0.86) instead of the default 0.5.
6. *Why XGBoost?* → Best-in-class for tabular data, fast, small to deploy.
7. *What are V1–V28?* → 28 features of one transaction, anonymised via PCA for
   privacy.
8. *Why Lambda not a server / SageMaker endpoint?* → Serverless = pay per
   request, no idle cost, free tier. A hosted endpoint bills 24/7.
9. *What's a cold start?* → First request loads the model (slow once), then warm
   and fast.
10. *How is it monitored?* → Every prediction logged to DynamoDB; Streamlit
    dashboard shows totals and trends.

**Your real metrics (real Kaggle data):** recall **0.86**, precision **0.64** at
default threshold, ROC-AUC **0.985**; tuning lifted F1 from **0.73 → 0.85**.

**The honesty point that impresses:** the dataset's PCA transform was never
published, so you can't compute `V1–V28` for a brand-new real swipe — your API
scores transactions already in that format. In production, the bank owns that
transform and your Lambda drops straight into the "model" slot of their pipeline.

---

<a name="part-10"></a>
## Part 10 — Glossary of every term

- **API** — a way for programs to talk over the internet via a URL.
- **API Gateway** — AWS service giving your Lambda a public URL.
- **Boosting** — building many small models where each fixes the last one's
  mistakes (how XGBoost works).
- **Booster** — xgboost's native model object (no scikit-learn needed).
- **Class imbalance** — when one label is far rarer than the other (fraud <1%).
- **Cold start** — the slow first Lambda run that loads code + model.
- **Confusion matrix** — a 2×2 table of correct vs incorrect predictions.
- **CORS** — browser rule controlling which web pages may call an API.
- **DMatrix** — xgboost's internal input data format.
- **DynamoDB** — AWS's serverless database; we log predictions here.
- **F1 score** — single number balancing precision and recall.
- **Feature** — one input fact about a transaction (`X`).
- **IAM role** — an AWS identity granting permissions (Lambda → DynamoDB).
- **Label / target** — the answer being predicted (`y`, the `Class` column).
- **Lambda** — AWS service that runs code on demand (serverless).
- **Overfitting** — when a model memorises training data and fails on new data.
- **PCA** — maths that compresses/anonymises features while keeping patterns
  (makes `V1–V28`).
- **Precision** — of flagged frauds, how many were real.
- **Probability** — the model's 0–1 confidence that a transaction is fraud.
- **Recall** — of real frauds, how many were caught.
- **ROC-AUC** — overall score of how well the model separates the classes
  (0.5 = random, 1.0 = perfect).
- **Serverless** — no server to manage; pay only when code runs.
- **scale_pos_weight** — xgboost setting to weight the rare class more.
- **SMOTE** — a technique that invents synthetic minority-class examples.
- **Test set** — data held back from training to score the model honestly.
- **Threshold** — the probability cut-off that turns a score into yes/no.
- **XGBoost** — the gradient-boosted-trees model we use.

---

*You now know this project A to Z. Read Part 3 and Part 9 the night before your
viva, and keep the actual code open when you read Part 5. Good luck! 🍀*

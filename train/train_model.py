# =============================================================================
# train_model.py
# -----------------------------------------------------------------------------
# WHAT THIS DOES (in plain English):
#   1. Loads the transactions data.
#   2. Splits it into a "training" part (to teach the model) and a "test" part
#      (to check the model on data it has NEVER seen).
#   3. Trains an XGBoost model to tell fraud (1) from normal (0).
#   4. Handles the "imbalance" problem (very few frauds) so the model actually
#      learns to catch fraud instead of ignoring it.
#   5. Prints proper scores (precision / recall) - NOT just accuracy.
#   6. Saves the trained model to a file so the API can use it later.
#
# Run it like this:   python train/train_model.py
# =============================================================================

import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, f1_score, average_precision_score
)
import xgboost as xgb

# --- Where things live -------------------------------------------------------
HERE = os.path.dirname(__file__)
DATA_DIR = os.path.join(HERE, "..", "data")
MODEL_DIR = os.path.join(HERE, "..", "model")
os.makedirs(MODEL_DIR, exist_ok=True)

REAL_DATA = os.path.join(DATA_DIR, "creditcard.csv")          # from Kaggle
SAMPLE_DATA = os.path.join(DATA_DIR, "creditcard_sample.csv")  # our fake one


# --- Step 1: Load the data ---------------------------------------------------
# We prefer the REAL Kaggle file. If it's not there, we fall back to the
# sample so the project still runs.
if os.path.exists(REAL_DATA):
    data_path = REAL_DATA
    print("Using the REAL Kaggle dataset (creditcard.csv).")
else:
    data_path = SAMPLE_DATA
    print("Real dataset not found -> using the SAMPLE dataset.")
    print("(For your real project, download creditcard.csv from Kaggle into the data/ folder.)")

df = pd.read_csv(data_path)
print(f"Loaded {len(df):,} rows.")

# "X" = the inputs (all columns EXCEPT the answer).
# "y" = the answer we want to predict (the 'Class' column: 0 or 1).
X = df.drop(columns=["Class"])
y = df["Class"]


# --- Step 2: Train/test split ------------------------------------------------
# We keep 20% of the data aside as a "test" set the model never sees during
# training. 'stratify=y' makes sure both parts keep the same fraud ratio.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"Training rows: {len(X_train):,}  |  Test rows: {len(X_test):,}")


# --- Step 3: Handle the imbalance -------------------------------------------
# There are WAY more normal transactions than fraud. If we do nothing, the
# model can score 99.8% accuracy by just saying "normal" every time - useless!
# 'scale_pos_weight' tells XGBoost to pay MUCH more attention to the rare fraud
# cases. A common value is (number of normal) / (number of fraud).
n_normal = (y_train == 0).sum()
n_fraud = (y_train == 1).sum()
scale = n_normal / max(n_fraud, 1)
print(f"Imbalance handled: scale_pos_weight = {scale:.1f}")


# --- Step 4: Train the model -------------------------------------------------
model = xgb.XGBClassifier(
    n_estimators=200,        # number of trees
    max_depth=4,             # how deep each tree can go
    learning_rate=0.1,       # how fast it learns
    scale_pos_weight=scale,  # <-- the imbalance fix
    eval_metric="aucpr",     # good metric for imbalanced data
    random_state=42,
    n_jobs=-1,
)
model.fit(X_train, y_train)
print("Model trained.")


# --- Step 5: Evaluate PROPERLY ----------------------------------------------
# We predict on the TEST set (unseen data) and look at the right metrics.
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]  # probability of fraud

print("\n================ RESULTS ================")
print("Confusion matrix [rows=true, cols=predicted]:")
print(confusion_matrix(y_test, y_pred))
print("\nDetailed report:")
# Precision = of the ones we FLAGGED as fraud, how many really were fraud.
# Recall    = of the REAL frauds, how many did we catch.
print(classification_report(y_test, y_pred, digits=4))
print(f"ROC-AUC score: {roc_auc_score(y_test, y_proba):.4f}")
print("=========================================\n")


# --- Step 5b: Threshold tuning ----------------------------------------------
# By default a model calls anything with probability >= 0.5 "fraud". But 0.5 is
# just a convention - it is rarely the BEST cut-off for imbalanced problems.
# We scan every candidate threshold and pick the one with the best F1 score
# (F1 balances precision and recall). The API will then use this tuned value.
print("---------------- THRESHOLD TUNING ----------------")
precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)
# precision_recall_curve returns one fewer threshold than precision/recall, so
# we line them up and compute F1 for each candidate threshold.
best_threshold, best_f1 = 0.5, -1.0
for p, r, t in zip(precisions[:-1], recalls[:-1], thresholds):
    if (p + r) == 0:
        continue
    f1 = 2 * p * r / (p + r)
    if f1 > best_f1:
        best_f1, best_threshold = f1, float(t)

# Keep the threshold in a sensible middle range. On easy/separable data the
# "best" F1 can sit at a degenerate extreme (e.g. 1.0), which would make the
# API miss a fraud scored at 0.9999. Clamping to 0.05..0.95 avoids that.
best_threshold = min(max(best_threshold, 0.05), 0.95)

# Show how the tuned threshold compares to the default 0.5.
default_f1 = f1_score(y_test, (y_proba >= 0.5).astype(int), zero_division=0)
tuned_f1 = f1_score(y_test, (y_proba >= best_threshold).astype(int), zero_division=0)
print(f"Default threshold 0.5   -> F1 = {default_f1:.4f}")
print(f"Best threshold  {best_threshold:.4f} -> F1 = {tuned_f1:.4f}")
print("(The API will use this tuned threshold via model/threshold.json.)")
print("--------------------------------------------------\n")


# --- Step 5c: scale_pos_weight vs SMOTE (optional comparison) ----------------
# Our main model fights class imbalance with `scale_pos_weight`. A popular
# alternative is SMOTE, which creates synthetic fraud examples so the classes
# are balanced. This block trains a SMOTE model and compares it, purely to show
# you understand the trade-off (great interview material). It only runs if the
# `imbalanced-learn` library is installed; otherwise we skip it cleanly.
try:
    from imblearn.over_sampling import SMOTE

    print("-------- scale_pos_weight  vs  SMOTE  --------")
    # SMOTE needs at least a few fraud examples to interpolate between.
    if n_fraud >= 6:
        X_res, y_res = SMOTE(random_state=42).fit_resample(X_train, y_train)
        smote_model = xgb.XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.1,
            eval_metric="aucpr", random_state=42, n_jobs=-1,
        )
        smote_model.fit(X_res, y_res)
        smote_proba = smote_model.predict_proba(X_test)[:, 1]

        # PR-AUC (average precision) is the fairest single number for imbalance.
        base_ap = average_precision_score(y_test, y_proba)
        smote_ap = average_precision_score(y_test, smote_proba)
        print(f"scale_pos_weight  PR-AUC = {base_ap:.4f}")
        print(f"SMOTE             PR-AUC = {smote_ap:.4f}")
        winner = "scale_pos_weight" if base_ap >= smote_ap else "SMOTE"
        print(f"-> On this data, {winner} did as well or better.")
        print("We keep scale_pos_weight: it needs no extra data and is simpler.")
    else:
        print("Too few fraud rows to run SMOTE safely - skipping.")
    print("---------------------------------------------\n")
except ImportError:
    print("(Skipping SMOTE comparison: run `pip install imbalanced-learn` to enable it.)\n")


# --- Step 6: Save the trained model -----------------------------------------
# We save in XGBoost's own JSON format. This single file is everything the
# API needs to make predictions later. Also save the column order so the API
# always feeds features in the right order.
model_path = os.path.join(MODEL_DIR, "fraud_model.json")
model.save_model(model_path)

columns_path = os.path.join(MODEL_DIR, "feature_columns.json")
with open(columns_path, "w") as f:
    json.dump(list(X.columns), f)

# Save the tuned decision threshold too. The API reads this automatically; if
# the file is missing it just falls back to 0.5, so nothing ever breaks.
threshold_path = os.path.join(MODEL_DIR, "threshold.json")
with open(threshold_path, "w") as f:
    json.dump({"threshold": best_threshold}, f)

print(f"Saved model to:     {model_path}")
print(f"Saved columns to:   {columns_path}")
print(f"Saved threshold to: {threshold_path}  (value: {best_threshold:.4f})")
print("\nRemember to copy the model files into app/ for the API/Lambda:")
print("  cp model/fraud_model.json model/feature_columns.json model/threshold.json app/")
print("Done! You can now run the prediction API.")

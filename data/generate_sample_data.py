# =============================================================================
# generate_sample_data.py
# -----------------------------------------------------------------------------
# WHY THIS FILE EXISTS:
# The "real" dataset for this project is the Kaggle "Credit Card Fraud
# Detection" dataset (creditcard.csv). It is free, but you need a Kaggle
# account to download it. So that YOUR PROJECT RUNS EVEN BEFORE you download
# the real data, this script creates a small FAKE dataset that has the exact
# same shape (columns) as the real one.
#
# The real dataset has these columns:
#   Time, V1, V2, ..., V28, Amount, Class
#   - V1..V28 are anonymised numbers (the bank hid the real meaning for privacy)
#   - Amount   = the transaction amount
#   - Class    = 0 for normal, 1 for fraud   <-- this is what we predict
#
# IMPORTANT: This fake data is ONLY so the code runs. For your real project /
# resume, download the real creditcard.csv from Kaggle and put it in this
# same "data" folder. The training script will automatically use the real
# file if it finds it.
# =============================================================================

import numpy as np
import pandas as pd
import os

# A "seed" makes the random numbers the same every time we run.
# That way your results are repeatable (important to mention in interviews).
np.random.seed(42)

N_NORMAL = 9800   # number of normal transactions
N_FRAUD = 200     # number of fraud transactions (few! this is the "imbalance")

def make_rows(n, is_fraud):
    """Create n rows of fake transactions."""
    # 28 anonymised feature columns V1..V28.
    # We shift the fraud numbers a little so the model has a pattern to learn.
    shift = 2.0 if is_fraud else 0.0
    v = np.random.normal(loc=shift, scale=1.0, size=(n, 28))

    time = np.random.randint(0, 172792, size=n)          # seconds over 2 days
    amount = np.abs(np.random.normal(88, 250, size=n))   # transaction amount
    label = np.ones(n) if is_fraud else np.zeros(n)      # 1 = fraud, 0 = normal

    data = np.column_stack([time, v, amount, label])
    cols = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount", "Class"]
    return pd.DataFrame(data, columns=cols)

# Build the full table: normal + fraud rows stacked together, then shuffled.
normal = make_rows(N_NORMAL, is_fraud=False)
fraud = make_rows(N_FRAUD, is_fraud=True)
df = pd.concat([normal, fraud]).sample(frac=1, random_state=42).reset_index(drop=True)
df["Class"] = df["Class"].astype(int)

# Save next to this script.
out_path = os.path.join(os.path.dirname(__file__), "creditcard_sample.csv")
df.to_csv(out_path, index=False)

print(f"Created sample dataset: {out_path}")
print(f"Rows: {len(df)}  |  Fraud: {df['Class'].sum()}  |  Normal: {(df['Class']==0).sum()}")
print(f"Fraud is only {100*df['Class'].mean():.2f}% of the data  <-- this is why it's called 'imbalanced'")

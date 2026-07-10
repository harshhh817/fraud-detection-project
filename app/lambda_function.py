# =============================================================================
# lambda_function.py  --  THE AWS LAMBDA VERSION (and the shared "brain")
# -----------------------------------------------------------------------------
# This is the code that will run on AWS. When someone sends a transaction to
# your API Gateway URL, AWS runs the function called `lambda_handler` below.
#
# What it does:
#   1. Reads the transaction the user sent (as JSON).
#   2. Loads the trained model (once, and keeps it in memory for speed).
#   3. Makes a prediction: fraud or not, plus a probability.
#   4. (Optional) Logs the prediction to DynamoDB for a monitoring dashboard.
#   5. Sends the answer back as JSON.
#
# IMPORTANT (why this file matters twice):
#   The `predict()` function below is the ONE place prediction logic lives.
#   The local test server (app/local_api.py) IMPORTS predict() from here, so the
#   two entry points can never drift apart. Change the logic once, here.
#
# NOTE FOR BEGINNERS:
#   - You don't run this file yourself. AWS runs it for you.
#   - To TEST the same logic on your own laptop, use app/local_api.py instead.
# =============================================================================

import json
import os
import uuid
import time
import xgboost as xgb
import numpy as np

# --- Load the model ONE time (not on every request) --------------------------
# AWS keeps the function "warm" between requests, so loading once here makes
# later predictions very fast. This is a good detail to mention in interviews.
HERE = os.path.dirname(__file__)
MODEL_PATH = os.path.join(HERE, "fraud_model.json")
COLUMNS_PATH = os.path.join(HERE, "feature_columns.json")
THRESHOLD_PATH = os.path.join(HERE, "threshold.json")   # optional (see below)

_model = xgb.XGBClassifier()
_model.load_model(MODEL_PATH)

with open(COLUMNS_PATH) as f:
    FEATURE_COLUMNS = json.load(f)   # the exact order of inputs the model expects

# The decision threshold: "if fraud_probability >= THRESHOLD, call it fraud".
# By default we use the classic 0.5. If train_model.py saved a TUNED threshold
# (threshold.json), we use that instead - it can catch more fraud. If the file
# is missing, we simply fall back to 0.5, so nothing ever breaks.
def _load_threshold(default=0.5):
    try:
        with open(THRESHOLD_PATH) as f:
            return float(json.load(f)["threshold"])
    except Exception:
        return default

THRESHOLD = _load_threshold()


def predict(transaction: dict):
    """
    Take one transaction (a dictionary of feature -> value) and return the
    fraud decision. This is the SINGLE shared prediction function used by both
    the AWS Lambda handler and the local test server.
    """
    # Build the input row in the EXACT column order the model was trained on.
    # If a feature is missing from the request, we default it to 0.
    row = [float(transaction.get(col, 0)) for col in FEATURE_COLUMNS]
    X = np.array([row])

    proba = float(_model.predict_proba(X)[0][1])   # probability it is fraud (0..1)
    is_fraud = proba >= THRESHOLD                    # tuned (or default 0.5) cut-off

    return {
        "is_fraud": bool(is_fraud),
        "fraud_probability": round(proba, 4),
        "decision": "FRAUD - review this transaction" if is_fraud else "OK - looks normal",
    }


# --- (Optional) Monitoring: log each prediction to DynamoDB -------------------
# This is 100% OPTIONAL and FREE (DynamoDB's 25 GB free tier). It only runs on
# AWS when you set an environment variable telling it which table to use:
#     FRAUD_LOG_TABLE = fraud-predictions
# If that variable is not set (e.g. when testing locally), logging is skipped.
# It is wrapped in try/except so a logging problem can NEVER break a prediction.
def log_prediction_dynamodb(transaction: dict, result: dict):
    table_name = os.environ.get("FRAUD_LOG_TABLE")
    if not table_name:
        return  # logging turned off -> do nothing

    try:
        import boto3  # boto3 is pre-installed in the Lambda runtime
        table = boto3.resource("dynamodb").Table(table_name)
        table.put_item(Item={
            "id": str(uuid.uuid4()),            # unique row id (partition key)
            "timestamp": int(time.time()),      # when it happened (epoch seconds)
            "is_fraud": result["is_fraud"],
            # DynamoDB dislikes plain floats, so store probability as a string.
            "fraud_probability": str(result["fraud_probability"]),
            "amount": str(transaction.get("Amount", 0)),
        })
    except Exception as e:
        # Never let logging failure affect the user's response.
        print(f"[warn] could not log to DynamoDB: {e}")


def lambda_handler(event, context):
    """
    This is the entry point AWS calls. `event` contains the incoming request.
    """
    try:
        # API Gateway puts the JSON body as a STRING in event["body"].
        # When testing directly in the Lambda console, event might already be
        # the dict. We handle both cases.
        if isinstance(event, dict) and "body" in event and event["body"] is not None:
            body = json.loads(event["body"])
        else:
            body = event

        result = predict(body)
        log_prediction_dynamodb(body, result)   # optional; no-op unless configured

        return {
            "statusCode": 200,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*",   # lets the web page call it
            },
            "body": json.dumps(result),
        }
    except Exception as e:
        return {
            "statusCode": 400,
            "headers": {"Access-Control-Allow-Origin": "*"},
            "body": json.dumps({"error": str(e)}),
        }

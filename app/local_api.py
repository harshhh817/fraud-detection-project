# =============================================================================
# local_api.py  --  TEST THE PROJECT ON YOUR OWN LAPTOP (no AWS needed)
# -----------------------------------------------------------------------------
# This runs a tiny web server on your computer so you can test predictions
# BEFORE (or instead of) deploying to AWS.
#
# It reuses the EXACT SAME predict() function from lambda_function.py, so what
# you test here is guaranteed to be what runs on AWS. (No copy-pasted logic!)
#
# HOW TO RUN:
#   1. Train the model first:  python train/train_model.py
#   2. Copy the model files next to this app (the README explains this):
#        cp model/fraud_model.json model/feature_columns.json app/
#   3. Run:  python app/local_api.py
#   4. Open http://127.0.0.1:5001  in your browser (or use web/index.html).
# =============================================================================

import os
import sys
import csv
import time
from flask import Flask, request, jsonify

# Let Python import lambda_function.py, which lives in THIS same folder.
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# The one and only prediction function. Importing it also loads the model once.
from lambda_function import predict

# Where we log predictions locally so the monitoring dashboard has data to show
# even without AWS. This is just a plain CSV file in the data/ folder.
LOG_PATH = os.path.join(HERE, "..", "data", "prediction_log.csv")


def log_prediction_local(transaction: dict, result: dict):
    """Append one prediction to a local CSV (best-effort; never crashes the API)."""
    try:
        new_file = not os.path.exists(LOG_PATH)
        with open(LOG_PATH, "a", newline="") as f:
            writer = csv.writer(f)
            if new_file:
                writer.writerow(["timestamp", "is_fraud", "fraud_probability", "amount"])
            writer.writerow([
                int(time.time()),
                result["is_fraud"],
                result["fraud_probability"],
                transaction.get("Amount", 0),
            ])
    except Exception as e:
        print(f"[warn] could not write local log: {e}")


app = Flask(__name__)


@app.route("/")
def home():
    return (
        "<h3>Fraud Detection API is running.</h3>"
        "<p>Send a POST request with transaction JSON to <code>/predict</code>.</p>"
        "<p>Or open web/index.html in your browser to use the form.</p>"
    )


@app.route("/predict", methods=["POST"])
def predict_route():
    try:
        body = request.get_json(force=True)
        result = predict(body)
        log_prediction_local(body, result)   # feeds the local dashboard
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    # We use port 5001 because on newer Macs, port 5000 is already taken by
    # the built-in "AirPlay Receiver" service.
    app.run(host="0.0.0.0", port=5001, debug=False)

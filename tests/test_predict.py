# =============================================================================
# test_predict.py  --  UNIT TESTS for the shared predict() function
# -----------------------------------------------------------------------------
# These tests prove the model behaves the way we expect. They give you a solid
# answer in interviews to "how do you know your code works?".
#
# HOW TO RUN (from the project root):
#     pip install pytest          # one-time
#     python -m pytest -v         # runs every test below
#
# Requirements to pass:
#   - You must have trained the model (python train/train_model.py) and copied
#     the model files into app/ (cp model/*.json app/). The README covers this.
# =============================================================================

import os
import sys

# Make sure Python can import the app code no matter where pytest is launched.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "app"))

from lambda_function import predict, FEATURE_COLUMNS


# These are REAL rows from the Kaggle dataset: one genuine normal transaction
# and one genuine fraud, so the tests check the model against real behaviour.
NORMAL = {"Time": 68.0, "V1": 1.1569, "V2": 0.0372, "V3": 0.5568, "V4": 0.5195,
          "V5": -0.4798, "V6": -0.3527, "V7": -0.2225, "V8": 0.1582, "V9": 0.0113,
          "V10": 0.1056, "V11": 1.6121, "V12": 0.3545, "V13": -1.4345, "V14": 0.797,
          "V15": 0.7451, "V16": 0.2229, "V17": -0.2292, "V18": -0.3648, "V19": -0.2541,
          "V20": -0.2219, "V21": -0.1827, "V22": -0.6123, "V23": 0.1973, "V24": 0.1749,
          "V25": 0.0325, "V26": 0.0995, "V27": -0.0268, "V28": 0.0042, "Amount": 2.69}

FRAUD = {"Time": 7672.0, "V1": 0.7027, "V2": 2.4264, "V3": -5.2345, "V4": 4.4167,
         "V5": -2.1708, "V6": -2.6676, "V7": -3.8781, "V8": 0.9113, "V9": -0.1662,
         "V10": -5.0092, "V11": 4.6757, "V12": -8.1672, "V13": 0.6386, "V14": -6.7633,
         "V15": 1.2969, "V16": -3.8118, "V17": -3.7541, "V18": -1.0492, "V19": 1.5542,
         "V20": 0.4227, "V21": 0.5512, "V22": -0.0098, "V23": 0.7217, "V24": 0.4732,
         "V25": -1.9593, "V26": 0.3195, "V27": 0.6005, "V28": 0.1293, "Amount": 1.0}


def test_output_has_the_expected_shape():
    """Every prediction must return these three keys with sensible types."""
    result = predict(NORMAL)
    assert set(result.keys()) == {"is_fraud", "fraud_probability", "decision"}
    assert isinstance(result["is_fraud"], bool)
    assert isinstance(result["fraud_probability"], float)
    assert isinstance(result["decision"], str)


def test_probability_is_between_0_and_1():
    """A probability that is outside 0..1 would mean something is broken."""
    p = predict(FRAUD)["fraud_probability"]
    assert 0.0 <= p <= 1.0


def test_normal_transaction_is_not_flagged():
    """The obviously-normal example should NOT be called fraud."""
    result = predict(NORMAL)
    assert result["is_fraud"] is False


def test_fraud_transaction_is_flagged():
    """The obviously-fraud example SHOULD be called fraud, with high confidence."""
    result = predict(FRAUD)
    assert result["is_fraud"] is True
    assert result["fraud_probability"] > 0.5


def test_missing_features_default_to_zero():
    """
    The API must not crash if the caller leaves features out. Sending only an
    Amount should still work (the missing V-features default to 0).
    """
    result = predict({"Amount": 100})
    assert "is_fraud" in result  # it ran without error


def test_feature_columns_loaded():
    """Sanity check: the model expects the V1..V28 + Amount style columns."""
    assert len(FEATURE_COLUMNS) > 0
    assert "Amount" in FEATURE_COLUMNS

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


# A clearly-normal transaction: all the V-features are 0 (average behaviour).
NORMAL = {"Amount": 88, "V1": 0, "V2": 0, "V3": 0, "V4": 0}

# A clearly-fraud transaction: in our dataset fraud rows have V1..V28 shifted
# up by about +2, so we recreate that pattern here.
FRAUD = {f"V{i}": 2.2 for i in range(1, 29)}
FRAUD["Amount"] = 650


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

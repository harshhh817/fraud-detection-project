# =============================================================================
# dashboard.py  --  A tiny "fraud monitoring" dashboard (Streamlit)
# -----------------------------------------------------------------------------
# WHAT THIS IS:
#   A simple one-page dashboard showing the predictions your API has made:
#   how many transactions, how many were fraud, and a chart over time.
#
# WHERE THE DATA COMES FROM:
#   data/prediction_log.csv - the local API (app/local_api.py) adds one row
#   to this file every time it makes a prediction. Simple as that.
#   (On AWS you'd read from DynamoDB instead - see deploy/AWS_DEPLOYMENT.md.)
#
# HOW TO RUN:
#     pip install streamlit
#     # make a few predictions first (via web/index.html) so there's data, then:
#     streamlit run dashboard/dashboard.py
# =============================================================================

import os
import pandas as pd
import streamlit as st

# The log file lives in the data/ folder, next to this dashboard/ folder.
HERE = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = os.path.join(HERE, "..", "data", "prediction_log.csv")

st.set_page_config(page_title="Fraud Monitoring", page_icon="💳")
st.title("💳 Fraud Detection — Monitoring Dashboard")

# --- Step 1: load the data ---------------------------------------------------
if not os.path.exists(LOG_FILE):
    st.info(
        "No predictions logged yet.\n\n"
        "Start the API (`python app/local_api.py`), open `web/index.html`, "
        "check a few transactions, then refresh this page."
    )
    st.stop()

df = pd.read_csv(LOG_FILE)

# Turn the raw timestamp (seconds) into a real date/time for nicer charts.
df["time"] = pd.to_datetime(df["timestamp"], unit="s")

# --- Step 2: the headline numbers ---------------------------------------------
total = len(df)
frauds = int(df["is_fraud"].sum())

col1, col2, col3 = st.columns(3)
col1.metric("Transactions checked", total)
col2.metric("Flagged as fraud", frauds)
col3.metric("Fraud rate", f"{frauds / total * 100:.1f}%")

# --- Step 3: fraud over time ---------------------------------------------------
st.subheader("Fraud flags over time")
frauds_only = df[df["is_fraud"]]
if frauds_only.empty:
    st.write("No fraud flagged yet — try the FRAUD example on the web page.")
else:
    per_hour = frauds_only.set_index("time").resample("1h").size()
    st.bar_chart(per_hour)

# --- Step 4: the raw log, newest first ------------------------------------------
st.subheader("Recent predictions")
st.dataframe(df.sort_values("time", ascending=False).head(50))

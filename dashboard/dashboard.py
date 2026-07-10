# =============================================================================
# dashboard.py  --  Fraud monitoring dashboard (Streamlit)
# -----------------------------------------------------------------------------
# WHAT THIS IS:
#   A polished one-page dashboard showing the predictions your API has made:
#   headline numbers ("KPIs"), a chart of activity over time, a fraud vs normal
#   breakdown, and a table of recent predictions.
#
# WHERE THE DATA COMES FROM:
#   data/prediction_log.csv - the local API (app/local_api.py) adds one row to
#   this file every time it makes a prediction. (On AWS you would read from the
#   DynamoDB table instead - see deploy/AWS_DEPLOYMENT.md.)
#
# HOW TO RUN:
#   pip install streamlit altair pandas
#   streamlit run dashboard/dashboard.py
#
# NOTE FOR BEGINNERS: most of the "design" is plain CSS injected once via
# st.markdown(). Streamlit draws the widgets; the CSS just makes them pretty.
# =============================================================================

import os
import pandas as pd
import altair as alt
import streamlit as st

# --- Page setup --------------------------------------------------------------
st.set_page_config(
    page_title="Fraud Monitoring",
    page_icon="💳",
    layout="wide",                     # use the full browser width
    initial_sidebar_state="collapsed",
)

# A small, brand-like colour palette we reuse everywhere.
INK = "#0f1c2e"        # near-black text
MUTED = "#6b7a90"      # grey text
FRAUD = "#e5484d"      # red  (fraud)
SAFE = "#12a594"       # teal (normal/ok)
ACCENT = "#3a6ff7"     # blue (accent)
CARD_BG = "#ffffff"
PAGE_BG = "#f4f6fb"

# --- Custom CSS (this is where the "design" lives) ---------------------------
st.markdown(f"""
<style>
  /* hide Streamlit's default chrome for a cleaner, app-like look */
  #MainMenu, header, footer {{visibility: hidden;}}
  .block-container {{padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1250px;}}
  .stApp {{background: {PAGE_BG};}}

  /* the top banner */
  .hero {{
    background: linear-gradient(120deg, #12233f 0%, #1f3b66 55%, #2f5aa8 100%);
    border-radius: 16px; padding: 22px 26px; color: #fff; margin-bottom: 18px;
  }}
  .hero h1 {{margin: 0; font-size: 26px; font-weight: 700; letter-spacing: .2px;}}
  .hero p  {{margin: 6px 0 0; font-size: 14px; opacity: .85;}}
  .pill {{
    display:inline-block; background: rgba(255,255,255,.16); color:#eaf1ff;
    padding: 3px 10px; border-radius: 999px; font-size: 12px; margin-top: 10px;
  }}

  /* KPI cards */
  .kpi {{
    background: {CARD_BG}; border-radius: 14px; padding: 16px 18px;
    box-shadow: 0 1px 3px rgba(16,28,46,.08); border: 1px solid #eef1f6;
    border-left: 5px solid {ACCENT}; height: 100%;
  }}
  .kpi .label {{color: {MUTED}; font-size: 13px; font-weight: 600; text-transform: uppercase; letter-spacing:.4px;}}
  .kpi .value {{color: {INK}; font-size: 30px; font-weight: 800; line-height: 1.15; margin-top: 4px;}}
  .kpi .sub   {{color: {MUTED}; font-size: 12px; margin-top: 2px;}}

  /* section headings + card wrappers for charts */
  .section-title {{font-size: 15px; font-weight: 700; color: {INK}; margin: 4px 0 8px;}}
  .panel {{background:{CARD_BG}; border:1px solid #eef1f6; border-radius:14px;
           padding:14px 16px; box-shadow:0 1px 3px rgba(16,28,46,.06);}}
</style>
""", unsafe_allow_html=True)


# --- Load the data -----------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = os.path.join(HERE, "..", "data", "prediction_log.csv")


def load_data():
    """Read the prediction log into a tidy DataFrame (or return an empty one)."""
    if not os.path.exists(LOG_FILE):
        return pd.DataFrame()
    df = pd.read_csv(LOG_FILE)
    # Make the columns clean and predictable no matter how they were written.
    df["is_fraud"] = df["is_fraud"].astype(str).str.lower().isin(["true", "1"])
    df["fraud_probability"] = pd.to_numeric(df["fraud_probability"], errors="coerce")
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df["time"] = pd.to_datetime(df["timestamp"], unit="s")
    return df.sort_values("time")


df = load_data()

# --- Header banner -----------------------------------------------------------
st.markdown("""
<div class="hero">
  <h1>💳 Fraud Detection — Monitoring Dashboard</h1>
  <p>Live view of every transaction scored by the model, with fraud flagged in real time.</p>
  <span class="pill">● data source: prediction log</span>
</div>
""", unsafe_allow_html=True)

# If there is no data yet, show a friendly message and stop.
if df.empty:
    st.info(
        "No predictions logged yet.\n\n"
        "Start the API (`python app/local_api.py`), open the web page, check a "
        "few transactions, then refresh this page."
    )
    st.stop()

# --- Compute the headline numbers -------------------------------------------
total = len(df)
frauds = int(df["is_fraud"].sum())
fraud_rate = frauds / total * 100 if total else 0
flagged_value = df.loc[df["is_fraud"], "amount"].sum()   # money on flagged txns


def kpi(col, label, value, sub, colour):
    """Render one coloured KPI card into a Streamlit column."""
    col.markdown(
        f"""<div class="kpi" style="border-left-color:{colour}">
              <div class="label">{label}</div>
              <div class="value">{value}</div>
              <div class="sub">{sub}</div>
            </div>""",
        unsafe_allow_html=True,
    )


c1, c2, c3, c4 = st.columns(4)
kpi(c1, "Transactions", f"{total:,}", "scored by the model", ACCENT)
kpi(c2, "Fraud flagged", f"{frauds:,}", "predicted as fraud", FRAUD)
kpi(c3, "Fraud rate", f"{fraud_rate:.1f}%", "of all transactions", "#b5851f")
kpi(c4, "Flagged amount", f"${flagged_value:,.0f}", "value on flagged txns", SAFE)

st.write("")   # a little spacing

# --- Charts row: activity over time  +  fraud/normal breakdown ---------------
left, right = st.columns([2, 1])

# Left: stacked bars of predictions per hour, split into normal vs fraud.
with left:
    st.markdown('<div class="section-title">Activity over time (per hour)</div>', unsafe_allow_html=True)
    per_hour = (
        df.assign(hour=df["time"].dt.floor("h"),
                  kind=df["is_fraud"].map({True: "Fraud", False: "Normal"}))
          .groupby(["hour", "kind"]).size().reset_index(name="count")
    )
    bars = (
        alt.Chart(per_hour)
        .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
        .encode(
            x=alt.X("hour:T", title=None, axis=alt.Axis(format="%b %d %H:%M", labelColor=MUTED)),
            y=alt.Y("count:Q", title="transactions", axis=alt.Axis(labelColor=MUTED, titleColor=MUTED)),
            color=alt.Color("kind:N",
                            scale=alt.Scale(domain=["Normal", "Fraud"], range=[SAFE, FRAUD]),
                            legend=alt.Legend(title=None, orient="top")),
            tooltip=["hour:T", "kind:N", "count:Q"],
        )
        .properties(height=300, background="transparent")
        .configure_view(strokeWidth=0)
        .configure_axis(grid=True, gridColor="#eef1f6", domainColor="#dfe4ec")
        .configure_legend(labelColor=MUTED)
    )
    st.altair_chart(bars, use_container_width=True)

# Right: a donut of fraud vs normal share.
with right:
    st.markdown('<div class="section-title">Fraud vs normal</div>', unsafe_allow_html=True)
    breakdown = pd.DataFrame({
        "kind": ["Normal", "Fraud"],
        "count": [total - frauds, frauds],
    })
    donut = (
        alt.Chart(breakdown)
        .mark_arc(innerRadius=62, cornerRadius=3)
        .encode(
            theta="count:Q",
            color=alt.Color("kind:N",
                            scale=alt.Scale(domain=["Normal", "Fraud"], range=[SAFE, FRAUD]),
                            legend=alt.Legend(title=None, orient="bottom")),
            tooltip=["kind:N", "count:Q"],
        )
        .properties(height=300, background="transparent")
        .configure_view(strokeWidth=0)
        .configure_legend(labelColor=MUTED)
    )
    st.altair_chart(donut, use_container_width=True)

# --- Recent predictions table (colour-coded) ---------------------------------
st.markdown('<div class="section-title">Recent predictions</div>', unsafe_allow_html=True)

recent = df.sort_values("time", ascending=False).head(25).copy()
recent["Time"] = recent["time"].dt.strftime("%b %d, %H:%M")
recent["Result"] = recent["is_fraud"].map({True: "🚨 FRAUD", False: "✅ OK"})
recent["Probability"] = (recent["fraud_probability"] * 100).round(2).astype(str) + "%"
recent["Amount"] = "$" + recent["amount"].round(2).astype(str)
table = recent[["Time", "Result", "Probability", "Amount"]]


def highlight_fraud(row):
    """Tint fraud rows red, normal rows faint teal."""
    is_fraud = row["Result"].startswith("🚨")
    colour = "background-color: #fdecec" if is_fraud else "background-color: #eafaf7"
    return [colour] * len(row)


st.dataframe(
    table.style.apply(highlight_fraud, axis=1),
    use_container_width=True,
    hide_index=True,
)

st.caption("Tip: run the API and check a few transactions, then refresh to see this update.")

# 🎤 Interview & Viva Guide

The questions you'll actually be asked about this project, with short, correct
answers in plain language. Read these until you can say them in your own words.
Don't memorise — understand.

---

## The 30‑second summary (practise saying this out loud)

"I built a machine‑learning system that detects fraudulent credit‑card
transactions. I trained an XGBoost model on transaction data where fraud is very
rare, so I handled the class imbalance and tuned the decision threshold so the
model actually catches fraud. Then I deployed the model as a serverless API
using AWS Lambda and API Gateway, so it returns a fraud/not‑fraud decision in
milliseconds. Every prediction is logged and shown on a monitoring dashboard,
and the prediction code is covered by unit tests. I kept it fully serverless so
there's no idle server cost."

---

## Core questions

**Q: What does the project do?**
It takes a transaction's details and predicts whether it's fraud, in real time,
through a web API.

**Q: What data did you use?**
The Kaggle "Credit Card Fraud Detection" dataset — about 284,000 real
transactions where the features are anonymised (V1–V28) for privacy, plus the
amount, and a label: 0 for normal, 1 for fraud. Only ~0.17% are fraud.

**Q: Why is that dataset hard?**
It's **imbalanced** — fraud is under 1% of the data. A lazy model can be 99.8%
accurate by always predicting "normal," while catching zero fraud. So accuracy
is misleading here.

**Q: How did you handle the imbalance?**
I used XGBoost's `scale_pos_weight` parameter, set to (normal count ÷ fraud
count). It makes the model weight the rare fraud cases much more heavily so it
learns to detect them. I also **compared it against SMOTE** (which creates
synthetic fraud examples) using PR‑AUC — on my data `scale_pos_weight` did as
well or better, and it's simpler because it doesn't invent artificial data.

**Q: Did you just use the default 0.5 threshold?**
No — 0.5 is only a convention. My training script scans every possible cut‑off
on the test set and picks the one with the best **F1 score** (the balance of
precision and recall), then saves it to a file the API reads automatically. If
the file is missing, the API safely falls back to 0.5.

**Q: How do you know your code works?**
I wrote **unit tests** (pytest) for the prediction function: a known‑normal
transaction must not be flagged, a known‑fraud one must be, probabilities must
stay in 0–1, and missing features must default to 0 instead of crashing. The
same `predict()` function is shared by the local server and the Lambda, so
testing it once covers both.

**Q: Why did you measure precision and recall instead of accuracy?**
- **Recall**: of all real frauds, how many I caught.
- **Precision**: of everything I flagged, how many were truly fraud.
Together they show real performance on the rare class. Accuracy hides failure on
imbalanced data.

**Q: What's the precision/recall trade‑off?**
If I lower the threshold to catch more fraud, recall goes up but I get more false
alarms, so precision drops. The right balance depends on business cost — missing
fraud vs. annoying honest customers.

**Q: Why XGBoost and not deep learning?**
The data is **tabular** (rows and columns). Gradient‑boosted trees like XGBoost
usually beat neural networks on tabular data, train fast, and are easy to deploy.
Deep learning shines on images/text, not this.

---

## Deployment / AWS questions

**Q: How is the model served?**
The trained model is saved as a small JSON file and loaded **inside an AWS Lambda
function**. API Gateway gives it a public URL. A request comes in → Lambda loads
the model → predicts → returns JSON.

**Q: Why Lambda instead of a SageMaker endpoint?**
A SageMaker hosted endpoint runs a server 24/7 and **bills per hour** even when
idle. Lambda is **serverless** — it only runs (and only bills) when a request
arrives, and it's free for the first million requests a month. So it's cheaper
and simpler for this scale. This is a deliberate cost decision.

**Q: Why not expose the SageMaker/model endpoint directly to users?**
You put API Gateway + Lambda in front so you can validate input, handle errors,
add auth later, and control access — you never expose the raw model.

**Q: What did it cost you?**
Effectively $0. Lambda, API Gateway (12 months), and DynamoDB free tiers cover a
student project easily, and I set a $1 AWS budget alert as a safety net.

**Q: How fast is a prediction?**
Milliseconds once the function is "warm," because the model loads once and stays
in memory between requests.

**Q: What is a "cold start"?**
The first request after idle has to load the function and model, so it's a bit
slower. After that it's fast. I could reduce it with provisioned concurrency, but
that's unnecessary for a demo.

---

## Monitoring questions

**Q: How do you monitor the model in production?**
Every prediction is logged — locally to a CSV file, and on AWS to a DynamoDB
table (free tier). A small **Streamlit dashboard** reads the log and shows
total transactions, how many were flagged, the fraud rate, and fraud over time.
That's how I'd spot problems like a sudden spike in fraud flags.

**Q: Could logging break the API?**
No — the logging call is wrapped in try/except, so if the database is down the
user still gets their prediction. Monitoring should never take down the thing
it monitors.

---

## "Make it better" questions (show you can think ahead)

**Q: How would you improve this?**
- Watch for **data drift** — retrain when transaction patterns change over time.
- Tune the threshold to the bank's **real money cost** of a missed fraud vs. a
  false alarm, instead of the statistical F1 balance I used.
- Add authentication (API keys / Cognito) so not just anyone can call it.
- Package the Lambda as a **container image** (I already wrote the Dockerfile)
  for a cleaner deploy, since XGBoost is large.

**Q: How would this scale to millions of users?**
Lambda scales automatically by running many copies in parallel, so it handles
spikes without me managing servers. That's a key benefit of serverless.

**Q: What are the risks / limitations?**
The public dataset is anonymised and a bit old, so real deployment would need
current data and constant retraining. Also, a model should assist human
reviewers, not auto‑block customers, to avoid harming legitimate users.

---

## If they ask you to draw the architecture

Draw this and talk through it:

```
User / Web form  ──▶  API Gateway  ──▶  Lambda (loads model, predicts)  ──▶  returns FRAUD / OK
                                              │
                                              └──▶  DynamoDB (logs)  ──▶  Dashboard
```

Say: "Serverless, pay‑per‑use, no idle cost, scales automatically."

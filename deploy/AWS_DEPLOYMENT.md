# 🚀 Deploy to AWS for FREE — Step by Step

This guide puts your model on the internet as a real API, at **zero cost**, by
running the model **inside AWS Lambda** (so there's no always‑on server to pay
for). Take your time; each step is small.

> **Before anything else — the safety net.** The moment your AWS account is
> ready, do Step 0. It guarantees you can never get a surprise bill.
> *(Using an AWS Academy account? Read the Academy section just below first —
> you can skip Step 0 entirely.)*

---

## Using an AWS Academy (Learner Lab) account? Read this first

Academy accounts work perfectly for this project and are even safer than a
personal account — there's no card attached, so you cannot be billed. Four
things are different from the steps below:

1. **SKIP Step 0** (the budget alert). The Billing console is blocked in
   Academy accounts, and you don't need it. Your lab page shows a credit
   meter instead; this project will use roughly $0 of it.

2. **The IAM role — the one gotcha that trips everyone.** Academy accounts
   can't create new IAM roles, and Lambda's "Create function" screen defaults
   to creating one. In Step 2, expand **"Change default execution role"** →
   choose **"Use an existing role"** → pick **`LabRole`**. Do this or the
   creation fails. Bonus: `LabRole` already includes DynamoDB permissions,
   so in Step 5 you can skip the "add the policy" part too.

3. **Click "Start Lab" first, every time.** Wait for the dot next to "AWS"
   to turn green, then click it to open the console. Sessions last ~4 hours,
   but everything you build (Lambda, API Gateway, DynamoDB) **persists
   between sessions** — you never rebuild. Being serverless, your API keeps
   answering even while your lab session is off; test it once from your phone.

4. **Stay in the `us-east-1` region.** Academy labs are locked to it. If a
   console page looks mysteriously empty, check the region picker (top-right)
   before panicking.

⚠️ **One real caveat:** Academy accounts are wiped when your course ends.
Before that date, record a short screen video of the live demo and screenshot
the Lambda + API Gateway consoles for your report — permanent proof it ran.

---

## Step 0 — Set a $1 budget alert (2 minutes, do this first)

1. Sign in to the AWS Console.
2. Search **"Budgets"** → **Create budget** → choose **Zero spend budget**
   (or a **Cost budget** set to **$1**).
3. Enter your email. Save.

Now AWS emails you the instant anything starts to cost money. This is your
insurance policy. With the Lambda approach below you should stay at $0 anyway.

---

## Why this stays free

| Service        | Free allowance                          | Do you exceed it? |
|----------------|------------------------------------------|-------------------|
| **Lambda**     | 1,000,000 requests/month — always free   | No (a demo uses a handful) |
| **API Gateway**| 1,000,000 requests/month for 12 months   | No |
| **DynamoDB**   | 25 GB storage — always free              | No |
| **SageMaker hosted endpoint** | ❌ bills per hour — WE DO NOT USE IT | We avoid it on purpose |

The one thing that costs money in these projects is a SageMaker **hosted
endpoint**. We deliberately don't use one — the model lives in Lambda instead.
That's both free *and* a better "I care about cost" story in interviews.

---

## Step 1 — Build the deployment package (the model + code in one zip)

Lambda needs your code, the model file, and the libraries bundled together.
XGBoost is a bit large, so the clean way is a **container image** OR a **Lambda
layer**. The simplest beginner path is a **zip with the model + a layer for
xgboost**. Here's the straightforward zip approach:

On your computer, from the project folder:

```bash
# 1) Make a build folder and install the libraries INTO it
mkdir -p build
pip install xgboost numpy -t build/

# 2) Copy your code + model into the same folder
cp app/lambda_function.py build/
cp model/fraud_model.json build/
cp model/feature_columns.json build/

# 3) Zip everything (the contents, not the folder itself)
cd build && zip -r ../fraud_lambda.zip . && cd ..
```

You now have `fraud_lambda.zip`.

> If the zip is bigger than 50 MB, upload it via an **S3 bucket** instead of
> directly (the console will offer this). Or use the **container image**
> option, which Lambda supports up to 10 GB — see the note at the bottom.

---

## Step 2 — Create the Lambda function

1. Console → **Lambda** → **Create function** → **Author from scratch**.
2. Name: `fraud-detection`. Runtime: **Python 3.12**. Architecture: leave default.
   *(AWS Academy account: expand "Change default execution role" → "Use an
   existing role" → `LabRole` — see the Academy section at the top.)*
3. Create the function.
4. In **Code** → **Upload from** → **.zip file** (or **Amazon S3** if large) →
   upload `fraud_lambda.zip`.
5. In **Runtime settings** → **Handler**, set it to:
   `lambda_function.lambda_handler`
6. In **Configuration → General configuration → Edit**: set **Memory** to
   512 MB and **Timeout** to 30 seconds (model loading needs a little room).
   Save.

**Test it:** open the **Test** tab, create an event with this JSON, and run:
```json
{ "V1": 2.2, "V2": 2.2, "V3": 2.2, "V4": 2.2, "Amount": 650 }
```
You should get back `"is_fraud": true`.

---

## Step 3 — Put an API in front of it (API Gateway)

1. Console → **API Gateway** → **Create API** → **HTTP API** → **Build**.
2. **Add integration** → **Lambda** → pick your `fraud-detection` function.
3. Give the API a name → **Next**.
4. **Routes**: method **POST**, path **/predict**. **Next** → **Create**.
5. After it's created, copy the **Invoke URL** (looks like
   `https://abc123.execute-api.us-east-1.amazonaws.com`). Your endpoint is that
   URL + `/predict`.

**Test from your terminal:**
```bash
curl -X POST https://YOUR-URL/predict \
  -H "Content-Type: application/json" \
  -d '{"V1":2.2,"V2":2.2,"V3":2.2,"V4":2.2,"Amount":650}'
```

---

## Step 4 — Connect the web page

Open `web/index.html`, paste your `https://YOUR-URL/predict` into the **API URL**
box, and click **Check**. Your model is now live on the internet. 🎉

---

## Step 5 (optional) — Log predictions to DynamoDB for a dashboard

Good news: the logging code is **already built into `lambda_function.py`**. It
stays switched off until you point it at a table with an environment variable,
so you only need to click a few things:

1. Console → **DynamoDB** → **Create table**. Name: `fraud-predictions`,
   partition key: `id` (String). Create.
2. Give your Lambda permission: **Configuration → Permissions → Role** → add the
   `AmazonDynamoDBFullAccess` policy (fine for a student project).
3. Turn logging on: **Configuration → Environment variables → Edit → Add** →
   key `FRAUD_LOG_TABLE`, value `fraud-predictions`. Save. That's it — every
   prediction now writes a row to the table. (`boto3` is already on Lambda.)
4. See it as a dashboard: run `streamlit run dashboard/dashboard.py` on your
   laptop with the same `FRAUD_LOG_TABLE` set (and AWS credentials configured),
   or just use the local CSV log the dashboard falls back to.

This step adds the "monitoring" story to your project, which interviewers love.

---

## Cleaning up (so you stay at $0)

Lambda, API Gateway, and DynamoDB don't charge at idle, so you can safely leave
them. If you ever created anything else, delete it from its console page. Your
$1 budget alert from Step 0 is watching your back either way.

---

## Note: the container‑image option (if the zip is too big)

If bundling xgboost makes the zip awkward, use a Lambda **container image**:
write a small `Dockerfile` starting `FROM public.ecr.aws/lambda/python:3.12`,
`pip install` your libs, `COPY` your code + model, push to **ECR**, and create
the Lambda from that image (up to 10 GB). This is still free within the Lambda
request allowance. Ask if you want the exact Dockerfile.

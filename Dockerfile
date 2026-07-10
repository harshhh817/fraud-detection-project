# =============================================================================
# Dockerfile  --  Package the model as an AWS Lambda CONTAINER IMAGE
# -----------------------------------------------------------------------------
# WHY THIS EXISTS:
#   XGBoost + numpy are fairly large. Zipping them can bump you over Lambda's
#   250 MB unzipped limit. A container image supports up to 10 GB, so it's the
#   clean, reliable way to ship this project. It's still FREE within Lambda's
#   1,000,000 free requests/month.
#
# HOW TO USE (summary - full steps are in deploy/AWS_DEPLOYMENT.md):
#   1. Build:   docker build -t fraud-detection .
#   2. Test locally (optional):
#        docker run -p 9000:8080 fraud-detection
#        curl "http://localhost:9000/2015-03-31/functions/function/invocations" \
#          -d '{"V1":2.2,"V2":2.2,"Amount":650}'
#   3. Push to Amazon ECR and create the Lambda from the image.
# =============================================================================

# Start from AWS's official Lambda base image for Python 3.12. It already knows
# how to run a lambda_handler for you.
FROM public.ecr.aws/lambda/python:3.12

# Install ONLY the libraries the prediction code needs at runtime (not Flask,
# not pytest - those are for local development only). This keeps the image lean.
RUN pip install --no-cache-dir xgboost numpy

# Copy the handler + the trained model + the column order + tuned threshold into
# the image's task root. ${LAMBDA_TASK_ROOT} is provided by the base image.
#
# BEFORE BUILDING, make sure these files exist in app/ (train first, then copy):
#   python train/train_model.py
#   cp model/fraud_model.json model/feature_columns.json model/threshold.json app/
COPY app/lambda_function.py ${LAMBDA_TASK_ROOT}/
COPY app/fraud_model.json ${LAMBDA_TASK_ROOT}/
COPY app/feature_columns.json ${LAMBDA_TASK_ROOT}/
COPY app/threshold.json ${LAMBDA_TASK_ROOT}/

# Tell Lambda which function to call for each request: file.function
CMD [ "lambda_function.lambda_handler" ]

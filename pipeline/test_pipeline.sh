#!/bin/bash

# Load environment variables from .env file
export $(grep -v '^#' ../.env | xargs)

# Define arguments
EXPERIMENT_NAME="FIRST_TEST"
COMPUTE_NAME="TEST-COMPUTE-INSTANCE-PM"
DATA_NAME="TEST_DATASET"
ENVIRONMENT_NAME="TEST_ENV"

# Run the pipeline script
python run_pipeline.py \
    --experiment_name "FIRST_TEST" \
    --compute_name "TEST-COMPUTE-INSTANCE-PM" \
    --data_name "TEST_DATASET" \
    --environment_name "TEST_ENV"

python run_pipeline.py --experiment_name "FIRST_TEST" --compute_name "TEST-COMPUTE-INSTANCE-PM" --data_name "TEST_DATASET" --environment_name "TEST_ENV"

python run_pipeline.py --experiment_name "MLFLOW_DEPLOY" --compute_name "TEST-COMPUTE-INSTANCE-PM" --data_name "TEST_DATASET" --environment_name "TEST_ENV"

python run_pipeline.py --experiment_name "TRAIN-PIPELINE-3" --compute_name "TEST-COMPUTE-INSTANCE-PM" --data_name "TEST_DATASET" --environment_name "TEST_ENV"

python run_pipeline.py --experiment_name "batch-monitoring" --compute_name "workspace-2-compute" --data_name "houseprice" --environment_name "TEST_ENV"
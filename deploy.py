import os
from dotenv import load_dotenv

from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient, Input, Output, load_component
from azure.ai.ml.entities import (
    Environment,
    BatchEndpoint,
    ModelBatchDeployment,
    ModelBatchDeploymentSettings,
    PipelineComponentBatchDeployment,
    Model,
    AmlCompute,
    Data,
    BatchRetrySettings,
    CodeConfiguration,
)
from azure.ai.ml.constants import AssetTypes, BatchDeploymentOutputAction
from azure.ai.ml.dsl import pipeline

load_dotenv()

# authenticate
credential = DefaultAzureCredential()

try:
    ml_client = MLClient(
        credential=credential,
        subscription_id=os.getenv('SUBSCRIPTION_ID'),
        resource_group_name=os.getenv('RESOURCE_GROUP'),
        workspace_name=os.getenv('WORKSPACE_NAME')
    )

except Exception as ex:
    print("HERE IN THE EXCEPTION BLOCK")
    print(ex)


# Create Batch endpoint
endpoint = BatchEndpoint(
    name="test-batch-endpoint-3",
    description="Testing MLFlow model Batch endpoint"
)
ml_client.batch_endpoints.begin_create_or_update(endpoint).result()

# Retrieve Registered model
model_name = "TEST_MODEL_3"
model_version = '1'
model = ml_client.models.get(name=model_name, version=model_version)

# Retrieve environment
environment_name = "TEST_ENV"
environment_version = "7"
env_asset = ml_client.environments.get(name=environment_name, version=environment_version)

# Create Batch deployment
deployment = ModelBatchDeployment(
    name="test-batch-deploy-3",
    description="Testing MLFlow model Batch deployment",
    endpoint_name=endpoint.name,
    model=model,
    code_configuration=CodeConfiguration(
        code="src/scoring_script", scoring_script="run-batch.py"
    ),
    environment=env_asset,
    compute="test-compute-cluster",
    settings=ModelBatchDeploymentSettings(
        instance_count=1,
        max_concurrency_per_instance=1,
        mini_batch_size=10,
        output_action=BatchDeploymentOutputAction.APPEND_ROW,
        output_file_name="predictions.csv",
        retry_settings=BatchRetrySettings(max_retries=3, timeout=30),
        logging_level="info",
    ),
)

ml_client.batch_deployments.begin_create_or_update(deployment).result()

# Set deployment name to batch endpoint
endpoint = ml_client.batch_endpoints.get(endpoint.name)
endpoint.defaults.deployment_name = deployment.name
ml_client.batch_endpoints.begin_create_or_update(endpoint).result()


default_ds = ml_client.datastores.get_default()

# Test deployment
job = ml_client.batch_endpoints.invoke(
    endpoint_name=endpoint.name,
    deployment_name=deployment.name,
    experiment_name='batch-endpoint-job',
    inputs={
        "house_data": Input(
            path="TEST_DATASET@latest", type="uri_file"
        )
    },
    outputs={
        "score": Output(
            type="uri_file",
            path=f"{default_ds.id}/paths/deploy-output-3/batch-output",
        )
    },
)
ml_client.jobs.get(job.name)
ml_client.jobs.stream(job.name)

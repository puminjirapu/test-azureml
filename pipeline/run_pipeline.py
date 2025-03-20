# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse

from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes

from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient

from azure.ai.ml.entities import AmlCompute
from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import Environment
from azure.ai.ml.dsl import pipeline
from azure.ai.ml import Input, Output, command
from azure.ai.ml.constants import AssetTypes, InputOutputModes

import json
import yaml
from pathlib import Path
import os
from dotenv import load_dotenv

def parse_args():
    parser = argparse.ArgumentParser("Deploy Training Pipeline")
    parser.add_argument("--experiment_name", type=str, help="Experiment Name")
    parser.add_argument("--compute_name", type=str, help="Compute Cluster Name")
    parser.add_argument("--data_name", type=str, help="Data Asset Name")
    parser.add_argument("--environment_name", type=str, help="Registered Environment Name")

    return parser.parse_args()


def main():
    args = parse_args()
    print(args)
    load_dotenv()

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

    try:
        print(ml_client.compute.get(args.compute_name))
    except:
        print("No compute found")



    print(os.getcwd())
    print('current', os.listdir())

    # Create pipeline job

    # 1. Define components
    parent_dir = Path(__file__).resolve().parent.parent
    parent_dir = os.path.join(parent_dir, "src")
    
    prep_data = command( 
        name="prep_data",
        display_name="prep-data",
        code=os.path.join(parent_dir, "prep"),
        command="python prep.py \
                --raw_data ${{inputs.raw_data}} \
                --train_data ${{outputs.train_data}}  \
                --test_data ${{outputs.test_data}}",
        environment=args.environment_name+"@latest",
        inputs={
            "raw_data": Input(type="uri_file"),
            },
        outputs={
            "train_data": Output(type="uri_folder"),
            "test_data": Output(type="uri_folder"),
            }
    )

    train_model = command( 
        name="train_model",
        display_name="train-model",
        code=os.path.join(parent_dir, "train"),
        command="python train.py \
                --train_data ${{inputs.train_data}} \
                --model_output ${{outputs.model_output}}",
        environment=args.environment_name+"@latest",
        inputs={"train_data": Input(type="uri_folder")},
        outputs={"model_output": Output(type="uri_folder")}
    )

    evaluate_model = command(
        name="evaluate_model",
        display_name="evaluate-model",
        code=os.path.join(parent_dir, "evaluate"),
        command="python evaluate.py \
                --model_name ${{inputs.model_name}} \
                --model_input ${{inputs.model_input}} \
                --test_data ${{inputs.test_data}} \
                --evaluation_output ${{outputs.evaluation_output}}",
        environment=args.environment_name+"@latest",
        inputs={
            "model_name": Input(type="string"),
            "model_input": Input(type="uri_folder"),
            "test_data": Input(type="uri_folder")
            },
        outputs={
            "evaluation_output": Output(type="uri_folder")
            }
    )

    register_model = command(
        name="register_model",
        display_name="register-model",
        code=os.path.join(parent_dir, "register"),
        command="python register.py \
                --model_name ${{inputs.model_name}} \
                --model_path ${{inputs.model_path}} \
                --evaluation_output ${{inputs.evaluation_output}} \
                --model_info_output_path ${{outputs.model_info_output_path}}",
        environment=args.environment_name+"@latest",
        inputs={
            "model_name": Input(type="string"),
            "model_path": Input(type="uri_folder"),
            "evaluation_output": Input(type="uri_folder")
            },
        outputs={
            "model_info_output_path": Output(type="uri_folder")
            }
    )

    # 2. Construct pipeline
    @pipeline()
    def test_pipeline(raw_data):
        
        prep = prep_data(
            raw_data=raw_data,
        )

        train = train_model(
            train_data=prep.outputs.train_data
        )

        evaluate = evaluate_model(
            model_name=os.getenv("MODEL_NAME"),
            model_input=train.outputs.model_output,
            test_data=prep.outputs.test_data
        )


        register = register_model(
            model_name=os.getenv("MODEL_NAME"),
            model_path=train.outputs.model_output,
            evaluation_output=evaluate.outputs.evaluation_output
        )

        return {
            "pipeline_job_train_data": prep.outputs.train_data,
            "pipeline_job_test_data": prep.outputs.test_data,
            "pipeline_job_trained_model": train.outputs.model_output,
            "pipeline_job_score_report": evaluate.outputs.evaluation_output,
        }


    pipeline_job = test_pipeline(
        Input(path=args.data_name + "@latest", type="uri_file")
    )

    # set pipeline level compute
    pipeline_job.settings.default_compute = args.compute_name
    # set pipeline level datastore
    pipeline_job.settings.default_datastore = "workspaceblobstore"

    pipeline_job = ml_client.jobs.create_or_update(
        pipeline_job, experiment_name=args.experiment_name
    )

    pipeline_job
    ml_client.jobs.stream(pipeline_job.name)

    
if __name__ == "__main__":
    main()
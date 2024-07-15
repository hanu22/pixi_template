import argparse
import logging
import time
from typing import Literal

from azure.ai.ml import Input, MLClient, Output, command
from azure.ai.ml.constants import AssetTypes, InputOutputModes
from azure.identity import DefaultAzureCredential

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%d/%m/%y %H:%M:%S",
)


def train_model(
    subscription_id: str,
    resource_group_name: str,
    workspace_name: str,
    environment_name: str,
    pipeline_name: Literal["training", "inference", "features"],
    data_asset_name: str,
    compute_name: str,
    repo_name: str,
    job_number: str,
) -> command:
    """Trains a model in AML.

    Args:
        subscription_id (str): subscription id
        resource_group_name (str): resource group name
        workspace_name (str): workspace name
        environment_name (str): environment name
        pipeline_name (Literal["training", "inference", "features"]): pipeline name
        data_asset_name (str): data asset name
        compute_name (str): compute name
        repo_name (str): repository name
        job_number (str): job number

    Returns:
        command: command
    """
    ml_client = MLClient(
        credential=DefaultAzureCredential(),
        subscription_id=subscription_id,
        resource_group_name=resource_group_name,
        workspace_name=workspace_name,
    )

    this_environment_name = f"{environment_name}-{pipeline_name}"

    # Get the latest data asset's version
    data_asset_version = ml_client.data.get(
        name=data_asset_name,
        label="latest",
    ).version

    job_name = f"{repo_name}-{job_number}-training"

    logging.info(f"{job_name=}")

    # TODO: @latest notation is not working for data assets
    inputs = {
        "assets": Input(
            type=AssetTypes.URI_FOLDER,
            path=f"azureml:{data_asset_name}:{data_asset_version}",
            mode=InputOutputModes.RO_MOUNT,
        )
    }

    outputs = {
        "training": Output(
            type=AssetTypes.URI_FOLDER,
            mode=InputOutputModes.RW_MOUNT,
        ),
    }

    command_train = "python -m spacy train \
    ${{inputs.assets}}/config.cfg \
    --output ${{outputs.training}} \
    --paths.train ${{inputs.assets}}/train.spacy \
    --paths.dev ${{inputs.assets}}/dev.spacy \
    --gpu-id 0 \
    --verbose"

    job = command(
        name=job_name,
        command=command_train,
        inputs=inputs,
        outputs=outputs,
        environment=f"{this_environment_name}@latest",
        compute=compute_name,
        experiment_name=repo_name,
    )

    ml_client.jobs.create_or_update(job)

    while True:
        state = ml_client.jobs.get(name=job_name).status

        if state == "Completed":
            logging.info(f"{job_name=} is in {state=}.")
            break

        elif state == "Failed":
            raise Exception(f"{job_name=} is in {state=}.")

        else:
            logging.info(f"{job_name=} is in {state=}." f"Waiting 60 seconds...")
            time.sleep(60)

    return job


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-s",
        "--subscription_id",
        required=True,
        help="subscription id",
    )

    parser.add_argument(
        "-rg",
        "--resource_group_name",
        required=True,
        help="resource group name",
    )

    parser.add_argument(
        "-ws",
        "--workspace_name",
        required=True,
        help="workspace name",
    )

    parser.add_argument(
        "-env",
        "--environment_name",
        required=True,
        help="environment name",
    )

    parser.add_argument(
        "-p",
        "--pipeline_name",
        required=True,
        help="pipeline name",
    )

    parser.add_argument(
        "-da",
        "--data_asset_name",
        required=True,
        help="data asset name",
    )

    parser.add_argument(
        "-c",
        "--compute_name",
        required=True,
        help="compute name",
    )

    parser.add_argument(
        "-r",
        "--repo_name",
        required=True,
        help="repository name",
    )

    parser.add_argument(
        "-j",
        "--job_number",
        required=True,
        help="job number",
    )

    args = parser.parse_args()

    train_model(
        args.subscription_id,
        args.resource_group_name,
        args.workspace_name,
        args.environment_name,
        args.pipeline_name,
        args.data_asset_name,
        args.compute_name,
        args.repo_name,
        args.job_number,
    )

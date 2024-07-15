import argparse
import json
import logging
import os
from pathlib import Path

from azure.ai.ml import MLClient
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import Model
from azure.identity import DefaultAzureCredential
from azureml.fsspec import AzureMachineLearningFileSystem

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%d/%m/%y %H:%M:%S",
)


def create_or_update_model(
    subscription_id: str,
    resource_group_name: str,
    workspace_name: str,
    model_name: str,
    repo_name: str = None,
    job_number: str = None,
) -> Model:
    """Creates or updates AML model asset.

    If a job number is provided, the training job path is used.
    Otherwise, local folder where model is downloaded is used.

    Args:
        subscription_id (str): subscription id
        resource_group_name (str): resource group name
        workspace_name (str): workspace name
        model_name (str): model name
        repo_name (str): repository name. Defaults to None
        job_number (str): job number. Defaults to None

    Returns:
        Model: aml model
    """
    # Use model path from training job
    if repo_name and job_number:
        datastore_path = (
            f"azureml://subscriptions/{subscription_id}/"
            f"resourcegroups/{resource_group_name}/"
            f"workspaces/{workspace_name}/"
            "datastores/workspaceblobstore/paths"
        )

        file_system = AzureMachineLearningFileSystem(datastore_path)

        # Path to the best trained model folder relative to the datastore path
        trained_model_path = (
            f"{datastore_path}/azureml/"
            f"{repo_name}-{job_number}-training/training/model-best/"
        )

        # If an evaluation job was run
        # metrics_path = (
        #     f"/azureml/{repo_name}-{job_number}-training/"
        #     "evaluation/metrics.json"
        # )

        metrics_path = f"{trained_model_path}/meta.json"

        with file_system.open(metrics_path) as file:
            metrics = json.load(file)

    # Use local model path where model is downloaded
    else:
        trained_model_path = Path(
            os.getenv("BUILD_SOURCESDIRECTORY"),
            "model",
        )

        metrics_path = Path(trained_model_path, "meta.json")

        with metrics_path.open() as file:
            metrics = json.load(file)

    ml_client = MLClient(
        credential=DefaultAzureCredential(),
        subscription_id=subscription_id,
        resource_group_name=resource_group_name,
        workspace_name=workspace_name,
    )

    model = Model(
        path=trained_model_path,
        type=AssetTypes.CUSTOM_MODEL,
        name=model_name,
        stage="Development",
        tags=metrics,
    )

    model = ml_client.models.create_or_update(model)

    logging.info(f"{model_name=} created or updated. Using {model.version=}.")

    return model


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
        "-m",
        "--model_name",
        required=True,
        help="model name",
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

    create_or_update_model(
        args.subscription_id,
        args.resource_group_name,
        args.workspace_name,
        args.model_name,
        args.repo_name,
        args.job_number,
    )

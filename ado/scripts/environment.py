import argparse
import logging
from typing import Literal

from azure.ai.ml import MLClient
from azure.ai.ml.entities import Environment
from azure.identity import DefaultAzureCredential

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%d/%m/%y %H:%M:%S",
)


def create_or_update_environment(
    subscription_id: str,
    resource_group_name: str,
    workspace_name: str,
    environment_name: str,
    pipeline_name: Literal["training", "inference", "features"],
) -> Environment:
    """Creates or updates AML environment asset.

    Args:
        subscription_id (str): subscription id
        resource_group_name (str): resource group name
        workspace_name (str): workspace name
        environment_name (str): environment name
        pipeline_name (Literal["training", "inference", "features"]): pipeline name

    Returns:
        Environment: aml environment
    """
    ml_client = MLClient(
        credential=DefaultAzureCredential(),
        subscription_id=subscription_id,
        resource_group_name=resource_group_name,
        workspace_name=workspace_name,
    )

    this_environment_name = f"{environment_name}-{pipeline_name}"

    if pipeline_name == "training":
        image = "curated/acpt-pytorch-2.2-cuda12.1:latest"
        conda_yml = "conda_training.yml"

    elif pipeline_name == "inference":
        image = "curated/acpt-pytorch-2.2-cuda12.1:latest"
        conda_yml = "conda_inference.yml"

    elif pipeline_name == "features":
        image = ""
        conda_yml = "conda_features.yml"

    environment = Environment(
        name=this_environment_name,
        image=f"mcr.microsoft.com/azureml/{image}",
        conda_file=f"mlops/environment/{conda_yml}",
    )

    environment = ml_client.environments.create_or_update(environment)

    logging.info(
        f"{this_environment_name=} created or updated. Using {environment.version=}."
    )

    return environment


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

    args = parser.parse_args()

    create_or_update_environment(
        args.subscription_id,
        args.resource_group_name,
        args.workspace_name,
        args.environment_name,
        args.pipeline_name,
    )

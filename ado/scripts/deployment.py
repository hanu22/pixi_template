import argparse
import logging
import os
import time
from typing import Literal

from azure.ai.ml import MLClient
from azure.ai.ml.entities import (
    CodeConfiguration,
    ManagedOnlineDeployment,
    OnlineRequestSettings,
)
from azure.identity import DefaultAzureCredential

# Get environment variables from the build pipeline to inject into the deployment
TRANSLATOR_API_KEY = os.getenv("TRANSLATOR_API_KEY")

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%d/%m/%y %H:%M:%S",
)


def create_or_update_deployment(
    subscription_id: str,
    resource_group_name: str,
    workspace_name: str,
    model_name: str,
    environment_name: str,
    pipeline_name: Literal["training", "inference", "features"],
    endpoint_name: str,
    deployment_name: str,
    instance_type: str,
    instance_count: int,
    max_concurrency: int,
    scoring_directory: str,
    scoring_script: str,
    registry_name: str = None,
) -> ManagedOnlineDeployment:
    """Creates or updates AML Managed Deployment.

    Args:
        subscription_id (str): subscription id
        resource_group_name (str): resource group name
        workspace_name (str): workspace name
        model_name (str): model name
        environment_name (str): environment name
        pipeline_name (Literal["training", "inference", "features"]): pipeline name
        endpoint_name (str): endpoint name
        deployment_name (str): deployment name
        instance_type (str): instance type
        instance_count (int): instance count
        max_concurrency (int): max concurrent requests per instance
        scoring_directory (str): scoring directory
        scoring_script (str): scoring script
        registry_name (str): registry name. Defaults to None

    Returns:
        ManagedOnlineDeployment: managed online deployment
    """
    # Use registry to get assets
    if registry_name:
        ml_client = MLClient(
            credential=DefaultAzureCredential(),
            registry_name=registry_name,
            registry_location="northeurope",
        )
    # Use workspace to get assets
    else:
        ml_client = MLClient(
            credential=DefaultAzureCredential(),
            subscription_id=subscription_id,
            resource_group_name=resource_group_name,
            workspace_name=workspace_name,
        )

    model = ml_client.models.get(
        name=model_name,
        label="latest",
    )

    environment_name = f"{environment_name}-{pipeline_name}"

    environment = ml_client.environments.get(
        name=environment_name,
        label="latest",
    )

    request_settings = OnlineRequestSettings(
        max_concurrent_requests_per_instance=max_concurrency,
        request_timeout_ms=5000,
        max_queue_wait_ms=500,
    )

    environment_variables = {
        "WORKER_COUNT": max_concurrency,
        "TRANSLATOR_API_KEY": TRANSLATOR_API_KEY,
        "MODEL_NAME": model.name,
        "MODEL_VERSION": model.version,
        "ENVIRONMENT_NAME": environment.name,
        "ENVIRONMENT_VERSION": environment.version,
    }

    deployment = ManagedOnlineDeployment(
        name=deployment_name,
        endpoint_name=endpoint_name,
        model=model,
        environment=environment,
        code_configuration=CodeConfiguration(
            code=scoring_directory,
            scoring_script=scoring_script,
        ),
        request_settings=request_settings,
        environment_variables=environment_variables,
        app_insights_enabled=True,
        instance_type=instance_type,
        instance_count=instance_count,
    )

    ml_client = MLClient(
        credential=DefaultAzureCredential(),
        subscription_id=subscription_id,
        resource_group_name=resource_group_name,
        workspace_name=workspace_name,
    )

    result = ml_client.online_deployments.begin_create_or_update(deployment)

    # Wait for deployment to be provisioned successfully or failed
    while True:
        state = result.status()

        if state == "Succeeded":
            logging.info(f"{deployment_name=} is in {state=}.")
            break

        elif state == "Failed":
            raise Exception(f"{deployment_name=} is in {state=}.")

        else:
            logging.info(f"{deployment_name=} is in {state=}." f"Waiting 10 seconds...")
            time.sleep(10)

    return deployment


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
        "-e",
        "--endpoint_name",
        required=True,
        help="endpoint name",
    )

    parser.add_argument(
        "-d",
        "--deployment_name",
        required=True,
        help="deployment name",
    )

    parser.add_argument(
        "-it",
        "--instance_type",
        required=True,
        help="instance type",
    )

    parser.add_argument(
        "-ic",
        "--instance_count",
        required=True,
        help="instance count",
    )

    parser.add_argument(
        "-mc",
        "--max_concurrency",
        required=True,
        help="max concurrency",
    )

    parser.add_argument(
        "-sd",
        "--scoring_directory",
        required=True,
        help="scoring directory",
    )

    parser.add_argument(
        "-ss",
        "--scoring_script",
        required=True,
        help="scoring script",
    )

    parser.add_argument(
        "-rn",
        "--registry_name",
        required=True,
        help="registry name",
    )

    args = parser.parse_args()

    create_or_update_deployment(
        args.subscription_id,
        args.resource_group_name,
        args.workspace_name,
        args.model_name,
        args.environment_name,
        args.pipeline_name,
        args.endpoint_name,
        args.deployment_name,
        args.instance_type,
        args.instance_count,
        args.max_concurrency,
        args.scoring_directory,
        args.scoring_script,
        args.registry_name,
    )

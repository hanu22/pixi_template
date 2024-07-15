import argparse
import logging
from typing import Literal

from azure.ai.ml import MLClient
from azure.ai.ml.exceptions import ValidationException
from azure.core.exceptions import ResourceNotFoundError
from azure.identity import DefaultAzureCredential

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%d/%m/%y %H:%M:%S",
)


def register_shared_environment(
    subscription_id: str,
    resource_group_name: str,
    workspace_name: str,
    environment_name: str,
    pipeline_name: Literal["training", "inference", "features"],
    registry_name: str,
):
    """Registers AML environment asset in Registry.

    Args:
        subscription_id (str): subscription id
        resource_group_name (str): resource group name
        workspace_name (str): workspace name
        environment_name (str): environment name
        pipeline_name (Literal["training", "inference", "features"]): pipeline name
        registry_name (str): registry name
    """
    ml_client = MLClient(
        credential=DefaultAzureCredential(),
        subscription_id=subscription_id,
        resource_group_name=resource_group_name,
        workspace_name=workspace_name,
    )

    environment_name = f"{environment_name}-{pipeline_name}"

    # Get the latest environment
    environment_latest = ml_client.environments.get(
        name=environment_name,
        label="latest",
    )

    logging.info(f"Got {environment_latest.version=}.")

    try:
        ml_client.environments.share(
            name=environment_name,
            version=environment_latest.version,
            registry_name=registry_name,
            share_with_name=environment_name,
            share_with_version=environment_latest.version,
        )

        logging.info(f"{environment_name=} shared with Registry.")

    except ValidationException as e:
        logging.info(f"An exception occurred: {e}")

    except ResourceNotFoundError as e:
        logging.info(f"An exception occurred: {e}")


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
        "-rn",
        "--registry_name",
        required=True,
        help="registry name",
    )

    args = parser.parse_args()

    register_shared_environment(
        args.subscription_id,
        args.resource_group_name,
        args.workspace_name,
        args.environment_name,
        args.pipeline_name,
        args.registry_name,
    )

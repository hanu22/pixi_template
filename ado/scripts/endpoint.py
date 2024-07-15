import argparse
import logging
import time

from azure.ai.ml import MLClient
from azure.ai.ml.entities import ManagedOnlineEndpoint
from azure.core.exceptions import ResourceNotFoundError
from azure.identity import DefaultAzureCredential

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%d/%m/%y %H:%M:%S",
)


def create_or_update_endpoint(
    subscription_id: str,
    resource_group_name: str,
    workspace_name: str,
    endpoint_name: str,
) -> ManagedOnlineEndpoint:
    """Creates or updates AML Managed Endpoint.

    Args:
        subscription_id (str): subscription id
        resource_group_name (str): resource group name
        workspace_name (str): workspace name
        endpoint_name (str): endpoint name

    Returns:
        ManagedOnlineEndpoint: managed online endpoint
    """
    ml_client = MLClient(
        credential=DefaultAzureCredential(),
        subscription_id=subscription_id,
        resource_group_name=resource_group_name,
        workspace_name=workspace_name,
    )

    # Get existing traffic to reuse it in case the new deployment fails
    try:
        traffic = ml_client.online_endpoints.get(endpoint_name).traffic

    except ResourceNotFoundError:
        logging.info("No endpoint exists to get traffic.")
        traffic = None

    # If no traffic is provided, the endpoint will be created/updated with 0% traffic
    endpoint = ManagedOnlineEndpoint(
        name=endpoint_name,
        auth_mode="key",
        traffic=traffic,
        tags={"authors": "[G. Harris]"},
    )

    # Raises InvalidOverwriteRequest if immutable properties are modified
    result = ml_client.online_endpoints.begin_create_or_update(endpoint)

    # Wait for endpoint to be provisioned successfully or failed
    while True:
        state = result.status()

        if state == "Succeeded":
            logging.info(f"{endpoint_name=} is in {state=}.")
            break

        elif state == "Failed":
            raise Exception(f"{endpoint_name=} is in {state=}.")

        else:
            logging.info(f"{endpoint_name=} is in {state=}." f"Waiting 10 seconds...")
            time.sleep(10)

    return endpoint


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
        "-e",
        "--endpoint_name",
        required=True,
        help="endpoint name",
    )

    args = parser.parse_args()

    create_or_update_endpoint(
        args.subscription_id,
        args.resource_group_name,
        args.workspace_name,
        args.endpoint_name,
    )

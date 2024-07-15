import argparse
import logging

from azure.ai.ml import MLClient
from azure.ai.ml.entities import ManagedOnlineEndpoint
from azure.identity import DefaultAzureCredential

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%d/%m/%y %H:%M:%S",
)


def update_endpoint_traffic(
    subscription_id: str,
    resource_group_name: str,
    workspace_name: str,
    endpoint_name: str,
    deployment_name: str,
) -> ManagedOnlineEndpoint:
    """Updates AML Managed Endpoint traffic.

    Args:
        subscription_id (str): subscription id
        resource_group_name (str): resource group name
        workspace_name (str): workspace name
        endpoint_name (str): endpoint name
        deployment_name (str): deployment name

    Returns:
        ManagedOnlineEndpoint: managed online endpoint

    """
    ml_client = MLClient(
        credential=DefaultAzureCredential(),
        subscription_id=subscription_id,
        resource_group_name=resource_group_name,
        workspace_name=workspace_name,
    )

    endpoint = ml_client.online_endpoints.get(
        name=endpoint_name,
    )

    # Other deployment traffic will be set to 0%
    endpoint.traffic = {
        deployment_name: 100,
    }

    ml_client.online_endpoints.begin_create_or_update(endpoint)

    logging.info(f"{deployment_name=} is allocated 100% traffic.")

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

    parser.add_argument(
        "-d",
        "--deployment_name",
        required=True,
        help="deployment name",
    )

    args = parser.parse_args()

    update_endpoint_traffic(
        subscription_id=args.subscription_id,
        resource_group_name=args.resource_group_name,
        workspace_name=args.workspace_name,
        endpoint_name=args.endpoint_name,
        deployment_name=args.deployment_name,
    )

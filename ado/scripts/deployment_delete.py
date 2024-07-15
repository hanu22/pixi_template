import argparse
import logging

from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%d/%m/%y %H:%M:%S",
)


def delete_deployment(
    subscription_id: str,
    resource_group_name: str,
    workspace_name: str,
    endpoint_name: str,
    deployment_name: str,
) -> str:
    """Deletes AML Managed Deployment.

    Args:
        subscription_id (str): subscription id
        resource_group_name (str): resource group name
        workspace_name (str): workspace name
        endpoint_name (str): endpoint name
        deployment_name (str): deployment name
    """
    ml_client = MLClient(
        credential=DefaultAzureCredential(),
        subscription_id=subscription_id,
        resource_group_name=resource_group_name,
        workspace_name=workspace_name,
    )

    ml_client.online_deployments.begin_delete(
        name=deployment_name,
        endpoint_name=endpoint_name,
    )


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

    delete_deployment(
        args.subscription_id,
        args.resource_group_name,
        args.workspace_name,
        args.endpoint_name,
        args.deployment_name,
    )

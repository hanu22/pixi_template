import argparse
import logging

from azure.ai.ml import MLClient
from azure.core.exceptions import ResourceNotFoundError
from azure.identity import DefaultAzureCredential

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%d/%m/%y %H:%M:%S",
)


def get_deployment_names(
    subscription_id: str,
    resource_group_name: str,
    workspace_name: str,
    endpoint_name: str,
) -> tuple[str]:
    """Gets old & new deployment names of AML Managed Endpoint.

    Args:
        subscription_id (str): subscription id
        resource_group_name (str): resource group name
        workspace_name (str): workspace name
        endpoint_name (str): endpoint name

    Returns:
        tuple: old deployment name, new deployment name
    """
    ml_client = MLClient(
        credential=DefaultAzureCredential(),
        subscription_id=subscription_id,
        resource_group_name=resource_group_name,
        workspace_name=workspace_name,
    )

    try:
        endpoint = ml_client.online_endpoints.get(endpoint_name)

    except ResourceNotFoundError as e:
        logging.info(f"Endpoint not found: {e.message}")
        new_deployment_name = "blue"
        old_deployment_name = "green"

    else:
        traffic = endpoint.traffic

        if traffic:
            logging.info(f"Endpoint found with {traffic=}")
            old_deployment_name = max(traffic, key=traffic.get)

            reverse_names = {
                "blue": "green",
                "green": "blue",
            }

            new_deployment_name = reverse_names.get(old_deployment_name)

        else:
            logging.info("Endpoint found but no traffic exists.")
            old_deployment_name = "blue"
            new_deployment_name = "green"

    logging.info(f"{old_deployment_name=}, {new_deployment_name=}")

    return old_deployment_name, new_deployment_name


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

    names = get_deployment_names(
        args.subscription_id,
        args.resource_group_name,
        args.workspace_name,
        args.endpoint_name,
    )

    print(f"##vso[task.setvariable variable=oldDeploymentName;isOutput=true]{names[0]}")
    print(f"##vso[task.setvariable variable=newDeploymentName;isOutput=true]{names[1]}")

import argparse
import logging
import time

from azure.ai.ml import MLClient
from azure.ai.ml.entities import AmlCompute
from azure.identity import DefaultAzureCredential

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%d/%m/%y %H:%M:%S",
)


def create_or_update_compute(
    subscription_id: str,
    resource_group_name: str,
    workspace_name: str,
    compute_name: str,
    compute_size: str,
    compute_type: str,
    min_instances: int,
    max_instances: int,
    idle_seconds: int,
) -> AmlCompute:
    """Creates or updates AML compute.

    Args:
        subscription_id (str): subscription id
        resource_group_name (str): resource group name
        workspace_name (str): workspace name
        endpoint_name (str): endpoint name
        compute_name (str): compute name
        compute_size (str): compute size
        compute_type (str): compute type
        min_instances (int): min instances
        max_instances (int):  max instances
        idle_seconds (int): idle seconds before scale down

    Returns:
        AmlCompute: compute
    """
    ml_client = MLClient(
        credential=DefaultAzureCredential(),
        subscription_id=subscription_id,
        resource_group_name=resource_group_name,
        workspace_name=workspace_name,
    )

    compute = AmlCompute(
        name=compute_name,
        type=compute_type,
        size=compute_size,
        min_instances=min_instances,
        max_instances=max_instances,
        idle_time_before_scale_down=idle_seconds,
    )

    result = ml_client.compute.begin_create_or_update(compute)

    # Wait for compute to be provisioned successfully or failed
    while True:
        state = result.status()

        if state == "Succeeded":
            logging.info(f"{compute_name=} is in {state=}.")
            break

        elif state == "Failed":
            raise Exception(f"{compute_name=} is in {state=}.")

        else:
            logging.info(f"{compute_name=} is in {state=}." f"Waiting 10 seconds...")
            time.sleep(10)

    return compute


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
        "-c",
        "--compute_name",
        type=str,
        required=True,
    )

    parser.add_argument(
        "-cs",
        "--compute_size",
        type=str,
        required=True,
    )

    parser.add_argument(
        "-ct",
        "--compute_type",
        type=str,
        required=True,
    )

    parser.add_argument(
        "-mii",
        "--min_instances",
        type=int,
        required=True,
    )

    parser.add_argument(
        "-mai",
        "--max_instances",
        type=int,
        required=True,
    )

    parser.add_argument(
        "-idle",
        "--idle_seconds",
        type=int,
        required=True,
    )

    args = parser.parse_args()

    create_or_update_compute(
        args.subscription_id,
        args.resource_group_name,
        args.workspace_name,
        args.compute_name,
        args.compute_size,
        args.compute_type,
        args.min_instances,
        args.max_instances,
        args.idle_seconds,
    )

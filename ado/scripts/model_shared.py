import argparse
import logging

from azure.ai.ml import MLClient
from azure.ai.ml.exceptions import ValidationException
from azure.core.exceptions import ResourceNotFoundError
from azure.identity import DefaultAzureCredential

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%d/%m/%y %H:%M:%S",
)


def register_shared_model(
    subscription_id: str,
    resource_group_name: str,
    workspace_name: str,
    model_name: str,
    registry_name: str,
):
    """Registers AML model asset in Registry.

    Args:
        subscription_id (str): subscription id
        resource_group_name (str): resource group name
        workspace_name (str): workspace name
        model_name (str): model name
        registry_name (str): registry name
    """
    ml_client = MLClient(
        credential=DefaultAzureCredential(),
        subscription_id=subscription_id,
        resource_group_name=resource_group_name,
        workspace_name=workspace_name,
    )

    # Get the latest model
    model_latest = ml_client.models.get(
        name=model_name,
        label="latest",
    )

    logging.info(f"Got {model_latest.version=}.")

    # stage attribute is set to "Development" when first registered
    model_latest.stage = "Production"

    try:
        ml_client.models.share(
            name=model_name,
            version=model_latest.version,
            registry_name=registry_name,
            share_with_name=model_name,
            share_with_version=model_latest.version,
        )

        logging.info(f"{model_name=} shared with Registry.")

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
        "-m",
        "--model_name",
        required=True,
        help="model name",
    )

    parser.add_argument(
        "-rn",
        "--registry_name",
        required=True,
        help="registry name",
    )

    args = parser.parse_args()

    register_shared_model(
        args.subscription_id,
        args.resource_group_name,
        args.workspace_name,
        args.model_name,
        args.registry_name,
    )

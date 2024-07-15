import argparse
import logging
from pathlib import Path

from azure.ai.ml import MLClient
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import Data
from azure.identity import DefaultAzureCredential

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%d/%m/%y %H:%M:%S",
)


def create_or_update_data_asset(
    subscription_id: str,
    resource_group_name: str,
    workspace_name: str,
    data_asset_name: str,
) -> Data:
    """Creates or updates AML data asset.

    Args:
        subscription_id (str): subscription id
        resource_group_name (str): resource group name
        workspace_name (str): workspace name
        data_asset_name (str): data asset name

    Returns:
        Data: data asset
    """
    ml_client = MLClient(
        credential=DefaultAzureCredential(),
        subscription_id=subscription_id,
        resource_group_name=resource_group_name,
        workspace_name=workspace_name,
    )

    data = Data(
        path=Path("mlops", "training", "assets"),
        type=AssetTypes.URI_FOLDER,
        name=data_asset_name,
        description="Spacy corpus binary data asset.",
    )

    data = ml_client.data.create_or_update(data)

    logging.info(f"{data.name=} created or updated. Using {data.version=}.")

    return data


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
        "-da",
        "--data_asset_name",
        required=True,
        help="data asset name",
    )

    args = parser.parse_args()

    create_or_update_data_asset(
        args.subscription_id,
        args.resource_group_name,
        args.workspace_name,
        args.data_asset_name,
    )

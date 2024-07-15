import argparse
import logging
from pathlib import Path

from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azureml.fsspec import AzureMachineLearningFileSystem

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%d/%m/%y %H:%M:%S",
)


def download_model(
    subscription_id: str,
    resource_group_name: str,
    workspace_name: str,
    model_name: str,
    download_path: Path,
    repo_name: str = None,
    job_number: str = None,
) -> None:
    """Downloads AML model into a local path.

    If repo name and job number are provided, download from training job path.
    Otherwise, download latest version from AML model asset.

    Args:
        subscription_id (str): subscription id
        resource_group_name (str): resource group name
        workspace_name (str): workspace name
        model_name (str): model name
        download_path (Path): local download path
        repo_name (str): repository name. Defaults to None
        job_number (str): job number. Defaults to None
    """
    ml_client = MLClient(
        credential=DefaultAzureCredential(),
        subscription_id=subscription_id,
        resource_group_name=resource_group_name,
        workspace_name=workspace_name,
    )

    # Download from training job path
    if repo_name and job_number:
        datastore_path = (
            f"azureml://subscriptions/{subscription_id}/"
            f"resourcegroups/{resource_group_name}/"
            f"workspaces/{workspace_name}/"
            "datastores/workspaceblobstore/paths"
        )

        file_system = AzureMachineLearningFileSystem(datastore_path)

        # Path to the best trained model folder relative to the datastore path
        trained_model_path = (
            f"{datastore_path}/azureml/"
            f"{repo_name}-{job_number}-training/training/model-best/"
        )

        logging.info(f"Downloading model from {trained_model_path}...")

        # Download from remote path (rpath) to local path (lpath)
        file_system.download(
            rpath=trained_model_path,
            lpath=str(download_path),
            recursive=True,
            **{"overwrite": "MERGE_WITH_OVERWRITE"},
        )

    # Download latest version from AML model asset
    else:
        model_latest = ml_client.models.get(
            name=model_name,
            label="latest",
        )

        logging.info(f"Downloading {model_latest.version=}...")

        ml_client.models.download(
            name=model_name,
            version=model_latest.version,
            download_path=download_path,
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
        "-m",
        "--model_name",
        required=True,
        help="model name",
    )

    parser.add_argument(
        "-dp",
        "--download_path",
        required=True,
        help="Local download path",
    )

    parser.add_argument(
        "-r",
        "--repo_name",
        required=False,
        help="repository name",
    )

    parser.add_argument(
        "-j",
        "--job_number",
        required=False,
        help="job number",
    )

    args = parser.parse_args()

    download_model(
        args.subscription_id,
        args.resource_group_name,
        args.workspace_name,
        args.model_name,
        args.download_path,
        args.repo_name,
        args.job_number,
    )

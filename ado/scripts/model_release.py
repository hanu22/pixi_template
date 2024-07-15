import argparse
import json
import logging

from azureml.fsspec import AzureMachineLearningFileSystem

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%d/%m/%y %H:%M:%S",
)


def release_trained_model(
    subscription_id: str,
    resource_group_name: str,
    workspace_name: str,
    repo_name: str,
    job_number: str,
    threshold: float,
) -> bool:
    """Releases a trained model in training job path.

    Args:
        subscription_id (str): subscription id
        resource_group_name (str): resource group name
        workspace_name (str): workspace name
        repo_name (str): repository name
        job_number (str): job number
        threshold (float): release threshold

    Returns:
        bool: release status
    """
    datastore_path = (
        f"azureml://subscriptions/{subscription_id}/"
        f"resourcegroups/{resource_group_name}/"
        f"workspaces/{workspace_name}/"
        "datastores/workspaceblobstore/paths"
    )

    file_system = AzureMachineLearningFileSystem(datastore_path)

    # Path to the metrics file relative to the datastore path
    metrics_path = (
        "/azureml/"
        f"{repo_name}-{job_number}-training/"
        "training/model-best/meta.json"
    )

    try:
        with file_system.open(metrics_path) as file:
            metrics = json.load(file)

    except FileNotFoundError:
        logging.error(f"File not found: {metrics_path=}")
        return False

    except json.JSONDecodeError:
        logging.error(f"Error decoding JSON file: {metrics_path=}")
        return False

    else:
        logging.info(f"{metrics=}.")

        cats_micro_f = metrics["performance"].get("cats_micro_f")
        ents_f = metrics["performance"].get("ents_f")

        # Model is trained on cats and ners
        if cats_micro_f and ents_f:
            release_cats = cats_micro_f > threshold
            release_ners = ents_f > threshold
            release = release_cats and release_ners

        # Model is trained on cats
        elif cats_micro_f:
            release = cats_micro_f > threshold

        # Model is trained on ners
        elif ents_f:
            release = ents_f > threshold

        else:
            logging.error("Metrics not found.")
            return False

        return release


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
        "-r",
        "--repo_name",
        required=True,
        help="repository name",
    )

    parser.add_argument(
        "-j",
        "--job_number",
        required=True,
        help="job number",
    )

    parser.add_argument(
        "-th",
        "--threshold",
        type=float,
        required=True,
        help="threshold value for release",
    )

    args = parser.parse_args()

    release = release_trained_model(
        args.subscription_id,
        args.resource_group_name,
        args.workspace_name,
        args.repo_name,
        args.job_number,
        args.threshold,
    )

    logging.info(f"Result: {release}")

    if release:
        print(
            "##vso[task.setvariable variable=releaseTrainedModel;isOutput=true]True",
        )
    else:
        print(
            "##vso[task.setvariable variable=releaseTrainedModel;isOutput=true]False",
        )

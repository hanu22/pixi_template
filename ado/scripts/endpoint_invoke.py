import argparse
import json
import logging
from pathlib import Path

from azure.ai.ml import MLClient
from azure.ai.ml.exceptions import ValidationException
from azure.core.exceptions import (
    HttpResponseError,
    ResourceNotFoundError,
    ServiceResponseError,
)
from azure.identity import DefaultAzureCredential
from tenacity import (
    retry,
    retry_if_exception_message,
    stop_after_attempt,
    wait_fixed,
)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%d/%m/%y %H:%M:%S",
)


@retry(
    retry=retry_if_exception_message(match="timeout"),
    wait=wait_fixed(5),
    stop=stop_after_attempt(5),
)
def invoke_endpoint(
    subscription_id: str,
    resource_group_name: str,
    workspace_name: str,
    endpoint_name: str,
    deployment_name: str,
    integration_tests_path: str,
):
    """Invokes AML Managed Endpoint.

    Args:
        subscription_id (str): subscription id
        resource_group_name (str): resource group name
        workspace_name (str): workspace name
        endpoint_name (str): endpoint name
        deployment_name (str): deployment name
        integration_tests_path (str): input data
    """
    ml_client = MLClient(
        credential=DefaultAzureCredential(),
        subscription_id=subscription_id,
        resource_group_name=resource_group_name,
        workspace_name=workspace_name,
    )

    request_file = Path(
        integration_tests_path,
        "request.json",
    )

    try:
        logging.info("Invoking Endpoint...")
        response = ml_client.online_endpoints.invoke(
            endpoint_name=endpoint_name,
            deployment_name=deployment_name,
            request_file=request_file,
        )

    except ResourceNotFoundError as e:
        logging.info(f"Endpoint not found: {e.message}")
        release = False

    except ValidationException as e:
        logging.info(f"Deployment not found: {e.message}")
        release = False

    except ServiceResponseError as e:
        logging.info(f"Remote host error: {e.message}")
        release = False

    except HttpResponseError as e:
        logging.info(f"HTTP error occurred: {e.message}")

        # The above error might be returned from the Score script
        # In this case, it will have a code in the message
        if "Code" in e.message:
            release = True
        else:
            release = False

    else:
        try:
            response_json = json.loads(response)

        except json.JSONDecodeError as e:
            logging.info(f"Error decoding response: {e}")
            release = False

        else:
            logging.info(f"Invocation successful. Response data: {response_json}")
            release = True

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

    parser.add_argument(
        "-itp",
        "--integration_tests_path",
        required=True,
        help="integration test data path",
    )

    args = parser.parse_args()

    release = invoke_endpoint(
        args.subscription_id,
        args.resource_group_name,
        args.workspace_name,
        args.endpoint_name,
        args.deployment_name,
        args.integration_tests_path,
    )

    logging.info(f"Result: {release}")

    if release:
        print(
            "##vso[task.setvariable variable=releaseNewDeployment;isOutput=true]True",
        )
    else:
        print(
            "##vso[task.setvariable variable=releaseNewDeployment;isOutput=true]False",
        )

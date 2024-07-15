# %%
import json
from pathlib import Path

from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv

from ado.scripts.deployment import create_or_update_deployment
from ado.scripts.endpoint import create_or_update_endpoint
from ado.scripts.endpoint_traffic import update_endpoint_traffic
from ado.scripts.environment import create_or_update_environment

load_dotenv(".env")

# %%
ml_client = MLClient.from_config(
    path=Path(
        "mlops",
        ".aml",
        "config.json",
    ),
    credential=DefaultAzureCredential(),
)

subscription_id = ml_client.subscription_id
resource_group_name = ml_client.resource_group_name
workspace_name = ml_client.workspace_name

# %%
model_name = ""
environment_name = ""
endpoint_name = ""

scoring_directory = "mlops/inference/code"
scoring_script = "score.py"

pipeline_name = "inference"
deployment_name = "blue"
instance_type = "Standard_NC4as_T4_v3"
instance_count = 1
max_concurrency = 10

request_file = Path(
    "mlops",
    "inference",
    "tests",
    "requests",
    "request.json",
)

# %%
create_or_update_environment(
    subscription_id=subscription_id,
    resource_group_name=resource_group_name,
    workspace_name=workspace_name,
    environment_name=environment_name,
    pipeline_name=pipeline_name,
)

# %%
create_or_update_endpoint(
    subscription_id=subscription_id,
    resource_group_name=resource_group_name,
    workspace_name=workspace_name,
    endpoint_name=endpoint_name,
)

# %%
create_or_update_deployment(
    subscription_id=subscription_id,
    resource_group_name=resource_group_name,
    workspace_name=workspace_name,
    model_name=model_name,
    environment_name=environment_name,
    pipeline_name=pipeline_name,
    endpoint_name=endpoint_name,
    deployment_name=deployment_name,
    instance_type=instance_type,
    instance_count=instance_count,
    max_concurrency=max_concurrency,
    scoring_directory=scoring_directory,
    scoring_script=scoring_script,
    registry_name=None,
)

# %%
response = ml_client.online_endpoints.invoke(
    endpoint_name=endpoint_name,
    deployment_name=deployment_name,
    request_file=request_file,
)

response_json = json.loads(response)

# %%
update_endpoint_traffic(
    subscription_id=subscription_id,
    resource_group_name=resource_group_name,
    workspace_name=workspace_name,
    endpoint_name=endpoint_name,
    deployment_name=deployment_name,
)

# %%

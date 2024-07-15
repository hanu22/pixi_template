# %%
from pathlib import Path

from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv

from ado.scripts.data import create_or_update_data_asset
from ado.scripts.environment import create_or_update_environment
from ado.scripts.evaluate import evaluate_model
from ado.scripts.model import create_or_update_model
from ado.scripts.model_download import download_model
from ado.scripts.model_release import release_trained_model
from ado.scripts.train import train_model

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
data_asset_name = ""
environment_name = ""
model_name = ""
repo_name = ""
pipeline_name = "training"
job_number = "1"
# compute_name = "NC24ads-A100-v4"
compute_name = "NC6s-v3"
threshold = 0.8

# %%
data = create_or_update_data_asset(
    subscription_id=subscription_id,
    resource_group_name=resource_group_name,
    workspace_name=workspace_name,
    data_asset_name=data_asset_name,
)

# %%
environment = create_or_update_environment(
    subscription_id=subscription_id,
    resource_group_name=resource_group_name,
    workspace_name=workspace_name,
    environment_name=environment_name,
    pipeline_name=pipeline_name,
)

# %%
train_model(
    subscription_id=subscription_id,
    resource_group_name=resource_group_name,
    workspace_name=workspace_name,
    repo_name=repo_name,
    job_number=job_number,
    compute_name=compute_name,
    environment_name=environment_name,
    pipeline_name=pipeline_name,
    data_asset_name=data_asset_name,
)

# %%
evaluate_model(
    subscription_id=subscription_id,
    resource_group_name=resource_group_name,
    workspace_name=workspace_name,
    repo_name=repo_name,
    job_number=job_number,
    compute_name=compute_name,
    environment_name=environment_name,
    pipeline_name=pipeline_name,
    data_asset_name=data_asset_name,
)

# %%
release = release_trained_model(
    subscription_id=subscription_id,
    resource_group_name=resource_group_name,
    workspace_name=workspace_name,
    repo_name=repo_name,
    job_number=job_number,
    threshold=threshold,
)

# %%
# BUG: registry is not working when a model is registered from training job path
# As a workaround, download model from training job path then register it

if release:
    # Download model from training job path to local folder
    download_model(
        subscription_id=subscription_id,
        resource_group_name=resource_group_name,
        workspace_name=workspace_name,
        model_name=model_name,
        download_path=Path("mlops", "model"),
        repo_name=repo_name,
        job_number=job_number,
    )

    # Register downloaded model as AML model asset
    create_or_update_model(
        subscription_id=subscription_id,
        resource_group_name=resource_group_name,
        workspace_name=workspace_name,
        model_name=model_name,
        repo_name=None,
        job_number=None,
    )

# %%

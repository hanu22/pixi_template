# %%
import json
from pathlib import Path

import requests
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv

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

endpoint_name = ""
url = ml_client.online_endpoints.get(name=endpoint_name).scoring_uri
key = ml_client.online_endpoints.get_keys(name=endpoint_name).primary_key

headers = {
    "Content-Type": "application/json",
    "Authorization": "Bearer " + key,
    # "azureml-model-deployment": "blue",
}

# %%
request_file = Path(
    "mlops",
    "inference",
    "tests",
    "requests",
    "request.json",
)

with request_file.open("r") as f:
    data = json.load(f)

response = requests.post(
    url,
    json=data,
    headers=headers,
    timeout=10,
)

try:
    response.raise_for_status()

except requests.exceptions.HTTPError as e:
    print(e)

else:
    response_json = response.json()

# %%
response_file = Path(
    "mlops",
    "inference",
    "tests",
    "responses",
    "response.json",
)

with response_file.open("w") as f:
    json.dump(
        response_json,
        f,
        indent=4,
    )

# %%

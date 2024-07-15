import copy
import re
from pathlib import Path
from typing import Literal

import toml
import yaml

LOCAL_PYTHON = Path(".venv", "Scripts", "python.exe")

# Open pyproject.toml to get python version from string (float)
with Path("pyproject.toml").open("r") as f:
    pyproject = toml.load(f)

python_version = re.findall(
    r"\d+\.\d+",
    pyproject["project"]["requires-python"],
)[0]

# Build dictionary for conda.yml
conda = {
    "name": ".venv",
    "channels": [
        "conda-forge",
    ],
    "dependencies": [
        f"python={python_version}",
        "pip=24.0",
        {"pip": []},
    ],
}

# Get pinned dependencies from requirements.txt
# strip \n and empty lines, and remove lines with # inside
# build a dictionary of {package: version}
pinned_dependencies = {}
with Path("requirements.txt").open("r") as f:
    for line in f:
        if line.strip() and "#" not in line:
            items = line.strip().split("==")
            pinned_dependencies[items[0]] = items[1]


def create_conda_pinned_dependencies(dependencies: list[str]) -> dict:
    """Creates conda pinned dependencies.

    Args:
        dependencies (list[str]): dependencies from pyproject.toml

    Returns:
        dict: conda dictionary with pinned dependencies
    """
    conda_pinned = copy.deepcopy(conda)

    # Get dependencies from pyproject.toml
    # split by == or >= and get the package name
    # if package is in pinned_dependencies, append it to dictionary
    for dependency in dependencies:
        if "==" in dependency:
            package = dependency.split("==")[0]
        elif ">=" in dependency:
            package = dependency.split(">=")[0]
        else:
            package = dependency

        if package in pinned_dependencies:
            conda_pinned["dependencies"][2]["pip"].append(
                f"{package}=={pinned_dependencies[package]}"
            )

    return conda_pinned


def write_conda_pinned_dependencies(
    pipeline_name: Literal["training", "inference", "features"],
    conda_pinned: dict,
):
    """Writes conda pinned dependencies to YAML file.

    Args:
        pipeline_name (Literal["training", "inference", "features"]): pipeline name
        conda_pinned (dict): conda dictionary with pinned dependencies
    """
    with Path(
        "mlops",
        "environment",
        f"conda_{pipeline_name}.yml",
    ).open("w") as f:
        yaml.dump(
            data=conda_pinned,
            stream=f,
            sort_keys=False,
        )


def main():
    for pipeline_name in ["training", "inference", "features"]:
        dependencies = pyproject["project"]["optional-dependencies"][pipeline_name]
        conda_pinned = create_conda_pinned_dependencies(dependencies)
        write_conda_pinned_dependencies(pipeline_name, conda_pinned)


if __name__ == "__main__":
    main()

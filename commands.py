import argparse
import subprocess
from pathlib import Path

# Define the Python versions
GLOBAL_PYTHON = subprocess.check_output(
    [
        "py",
        "-3.9",
        "-c",
        "import sys; print(sys.executable)",
    ]
).strip()

LOCAL_PYTHON = Path(".venv", "Scripts", "python.exe")
LOCAL_PIP_COMPILE = Path(".venv", "Scripts", "pip-compile")
LOCAL_PIP_SYNC = Path(".venv", "Scripts", "pip-sync")
LOCAL_PRE_COMMIT = Path(".venv", "Scripts", "pre-commit")


def create():
    print("Creating .venv...")

    subprocess.run([GLOBAL_PYTHON, "-m", "venv", ".venv"])


def install():
    print("Installing dependencies...")

    subprocess.run(
        [
            LOCAL_PYTHON,
            "-m",
            "pip",
            "install",
            "--upgrade",
            "pip",
        ]
    )

    subprocess.run(
        [
            LOCAL_PYTHON,
            "-m",
            "pip",
            "install",
            "pip-tools",
            "--require-virtualenv",
        ]
    )

    subprocess.run(
        [
            LOCAL_PIP_COMPILE,
            "--output-file",
            "requirements.txt",
            "pyproject.toml",
            "--all-extras",
        ]
    )

    subprocess.run(
        [
            LOCAL_PIP_SYNC,
            "requirements.txt",
        ]
    )

    subprocess.run(
        [
            LOCAL_PYTHON,
            "conda.py",
        ]
    )


def hooks():
    print("Setting up pre-commit...")

    Path(".git", "hooks", "pre-commit").unlink(missing_ok=True)

    subprocess.run([LOCAL_PRE_COMMIT, "install"])
    subprocess.run([LOCAL_PRE_COMMIT, "autoupdate"])


def update():
    print("Updating dependencies...")

    subprocess.run(
        [
            LOCAL_PYTHON,
            "-m",
            "pip",
            "install",
            "--upgrade",
            "pip",
        ]
    )

    subprocess.run(
        [
            LOCAL_PIP_SYNC,
            "requirements.txt",
        ]
    )

    subprocess.run(
        [
            LOCAL_PYTHON,
            "conda.py",
        ]
    )


def check():
    print("Running checks...")
    subprocess.run([".venv\\Scripts\\ruff", "check", "--fix", "."])
    subprocess.run([".venv\\Scripts\\ruff", "format", "."])
    subprocess.run([".venv\\Scripts\\sqlfluff", "fix", "."])
    subprocess.run([".venv\\Scripts\\sqlfluff", "lint", "."])


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--command",
        "-c",
        type=str,
        choices=[
            "setup",
            "create",
            "install",
            "hooks",
            "update",
            "check",
            "clean",
        ],
        help="Command to run",
    )

    args = parser.parse_args()

    if args.command == "setup":
        create()
        install()
        hooks()

    elif args.command == "create":
        create()

    elif args.command == "install":
        install()

    elif args.command == "hooks":
        hooks()

    elif args.command == "update":
        update()

    elif args.command == "check":
        check()


if __name__ == "__main__":
    main()

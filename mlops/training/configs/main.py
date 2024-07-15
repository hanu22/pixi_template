import configparser
import subprocess
from pathlib import Path

LOCAL_PYTHON = Path(".venv", "Scripts", "python.exe")
OUTPUT_FILE = Path("mlops", "training", "assets", "config.cfg")


class CaseSensitiveConfigParser(configparser.ConfigParser):
    def optionxform(self, optionstr: str) -> str:
        """Override optionxform to prevent lowercasing of keys.

        By default ConfigParser lowercases keys

        Args:
            optionstr (str): input string

        Returns:
            str: string without lowercasing
        """
        return optionstr


def initialize_config(pipeline: str):
    print("Initializing config file...")
    subprocess.run(
        [
            LOCAL_PYTHON,
            "-m",
            "spacy",
            "init",
            "config",
            OUTPUT_FILE,
            "--lang",
            "en",
            "--pipeline",
            pipeline,
            "--optimize",
            "accuracy",
            "--gpu",
            "--force",
        ]
    )


def update_batch_size(batch_size: str):
    config = CaseSensitiveConfigParser()
    config.read(OUTPUT_FILE)

    config.set("nlp", "batch_size", batch_size)

    with OUTPUT_FILE.open("w") as file:
        config.write(file)


def add_train_augmenter():
    config = CaseSensitiveConfigParser()
    config.read(OUTPUT_FILE)

    config.remove_option("corpora.train", "augmenter")

    config["corpora.train.augmenter"] = {
        "@augmenters": "spacy.lower_case.v1",
        "level": "0.3",
    }

    with OUTPUT_FILE.open("w") as file:
        config.write(file)


def main():
    initialize_config("transformer,ner")
    initialize_config("transformer,textcat")
    initialize_config("transformer,textcat_multilabel")
    update_batch_size(batch_size="512")
    add_train_augmenter()


if __name__ == "__main__":
    main()

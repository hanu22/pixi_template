import json
from pathlib import Path


def read_mltable_annotations(files_path: Path) -> list[dict]:
    """Reads all annotation files in path and joins them.

    The labelling service downloads a number of JSONL files,
    each has a 1000 labelled images, using the MLTable annotation format.

    Args:
        files_path (Path): JSONL files full path.

    Returns:
        list[dict]: list of dicts, each for one image.
    """
    annotations = []

    # Get all jsonl files in path
    # These are downloaded with the format: labeledDatapoints_*.jsonl
    files = files_path.glob("labeledDatapoints_*.jsonl")

    # The file names always start from 1
    for file in files:
        with file.open("r") as f:
            # Each line is for one image, as a dict string
            for line in f:
                # Convert dict string to a dict object literal
                item = json.loads(line)
                annotations.append(item)

    return annotations

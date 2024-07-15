import json

import pandas as pd


def count_label_frequency_mltable(annotations: list[dict]) -> pd.DataFrame:
    """Counts the frequency of each label in annotations.

    Args:
        annotations (list[dict]): JSONL files in MLTable format.

    Returns:
        pd.DataFrame: dataframe of label frequencies
    """
    label_count = {}

    for annotation in annotations:
        for image_label in annotation["label"]:
            label = image_label["label"]
            if label not in label_count:
                label_count[label] = 0
            label_count[label] += 1

    df = pd.DataFrame(
        label_count.items(),
        columns=[
            "labels",
            "count",
        ],
    ).sort_values(
        by="count",
        ascending=False,
    )

    return df


def count_label_frequency_coco(annotations: json) -> pd.DataFrame:
    """Counts the frequency of each label in annotations.

    Args:
        annotations (json): COCO dataset JSON file.

    Returns:
        pd.DataFrame: dataframe of label frequencies
    """
    categories = {
        category["id"]: category["name"] for category in annotations["categories"]
    }
    annotations = annotations["annotations"]

    # Extract the labels and their frequencies
    labels = []

    for annotation in annotations:
        labels.append(
            {
                "image_id": annotation["image_id"],
                "category_id": annotation["category_id"],
                "category_name": categories[annotation["category_id"]],
            }
        )

    # Create a dataframe of the label frequencies
    df = pd.DataFrame(labels)
    df = (
        df.groupby("category_name")
        .count()
        .sort_values(
            by="image_id",
            ascending=False,
        )
        .rename(columns={"image_id": "count"})
        .drop(
            "category_id",
            axis=1,
        )
    )

    return df

import json


def filter_mltable_annotations(
    annotations: list[dict],
    categories: list[str],
) -> list[dict]:
    """Filters annotations to only include target labels.

    Args:
        annotations (list[dict]): Unfiltered MLTable data.
        categories (list[str]): List of target labels.

    Returns:
        list[dict]: list of dicts, each for one image.
    """
    annotations_filtered = []

    for annotation in annotations:
        # Filter the labels and their confidences
        labels_filtered = []
        confidences_filtered = []

        for idx, label in enumerate(annotation["label"]):
            if label["label"] in categories:
                labels_filtered.append(label)
                confidences_filtered.append(annotation["label_confidence"][idx])

        if labels_filtered:
            # Update the label and label_confidence
            # Only include target label objects and confidences
            annotation["label"] = labels_filtered
            annotation["label_confidence"] = confidences_filtered

            # Add the filtered JSON object to the result list
            annotations_filtered.append(annotation)

    return annotations_filtered


def filter_coco_annotations(
    annotations: json,
    categories: list[str],
) -> json:
    """Filters labels from a coco dataset to only keep given labels.

    Coco dataset format: https://cocodataset.org/#format-data

    Args:
        annotations (json): dataset in a COCO JSON format.
        categories (list[str]): list of labels to keep.

    Returns:
        json: filtered COCO dataset JSON file.
    """
    # Keep only target categories
    categories_filtered = [
        cat for cat in annotations["categories"] if cat["name"] in categories
    ]

    # Update the categories to only include target categories
    annotations["categories"] = categories_filtered

    # Create a set of IDs for the filtered categories
    cat_ids_filtered = set(cat["id"] for cat in categories_filtered)

    # Keep only target annotations
    annotations_filtered = [
        annotation
        for annotation in annotations["annotations"]
        if annotation["category_id"] in cat_ids_filtered
    ]

    # Update the annotations to only include target annotations
    annotations["annotations"] = annotations_filtered

    return annotations


def convert_coco_to_mltable(
    coco_annotations: json,
    image_prefix: str,
) -> json:
    """Converts COCO annotations file into an MLTable annotations.

    x_min and y_min coordinates correspond to the top left corner of the bounding box.

    The Azure MLTable (JSONL) format is as follows:

    {"image_url": "azureml://subscriptions/7df58d01-0877-4c17-8b30-125a47eafebc/
    resourcegroups/ds-computer-vision-dev-eun-rg/workspaces/
    ds-computer-vision-dev-eun-ws/datastores/workspaceblobstore/paths/UI/
    2023-05-11_165452_UTC/t-shots-jpgs/9265078_73.jpg",
    "label":
    [{"label": "fSC-Mix",
    "topX": 0.47419136418831936,
    "topY": 0.16295539770233497,
    "bottomX": 0.5199522782800796,
    "bottomY": 0.27442967561300324}],
    "label_confidence": [1.0],
    "image_width": 800,
    "image_height": 800}

    Args:
        coco_annotations (json): COCO dataset JSON file.
        image_prefix (str): Image location in Azure Machine learning datastore.

    Returns:
        json: COCO dataset in JSONL format.
    """
    # Loop through the images
    data_jsonl = []

    for item in coco_annotations["images"]:
        # Image number
        image_id = item["id"]

        # Get the image details
        height = item["height"]
        width = item["width"]

        # Split the image URL
        url_tokens = item["coco_url"].split("/")

        # Remove the first 3 tokens as they are not needed in the output URL
        url_tokens = url_tokens[3:]

        image_url = f"{image_prefix}" + "/".join(url_tokens)

        # Get the labels for this image
        labels = []
        for annotation in coco_annotations["annotations"]:
            if annotation["image_id"] == image_id:
                bbox = annotation["bbox"]
                x_min, y_min, bbox_width, bbox_height = bbox

                topX = x_min
                topY = y_min
                bottomX = x_min + bbox_width
                bottomY = y_min + bbox_height
                category_id = annotation["category_id"]

                category_name = None
                for category in coco_annotations["categories"]:
                    if category["id"] == category_id:
                        category_name = category["name"]
                        break

                label = {
                    "label": category_name,
                    "topX": topX,
                    "topY": topY,
                    "bottomX": bottomX,
                    "bottomY": bottomY,
                }

                labels.append(label)

        data = {
            "image_url": image_url,
            "label": labels,
            "label_confidence": [1.0] * len(labels),
            "image_width": width,
            "image_height": height,
        }

        data_jsonl.append(data)

    return data_jsonl

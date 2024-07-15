# %%
import json
from pathlib import Path

from label.azure.scripts.eda import count_label_frequency_mltable
from label.azure.scripts.load import read_mltable_annotations
from label.azure.scripts.preprocess import filter_mltable_annotations

# %%
# Open the labelled images JSONL files downloaded from Azure
annotations = read_mltable_annotations(
    files_path=Path(
        "data",
        "processed",
        "v1",
    )
)

# Get the top 80% objects
with Path(
    "data",
    "processed",
    "v1",
    "objects.txt",
).open("r") as f:
    categories = f.read().splitlines()

# Each image labelled in Azure will have multiple objects labelled
# Remove labelled objects that are not in the top 80% objects
# This to train/validate on the top 80% objects
annotations_filtered = filter_mltable_annotations(
    annotations=annotations,
    categories=categories,
)

# %%
# Count the frequency of objects in the top 80% of objects
df_labels = count_label_frequency_mltable(annotations=annotations_filtered)

# View sum of objects on all images
df_labels.sum()

# %%
# Save the filtered annotations
with open(
    Path(
        "data",
        "processed",
        "v1",
        "labeledDatapoints_filtered.jsonl",
    ),
    "w",
) as file:
    for annotations in annotations_filtered:
        # Convert dict Object literal into a dict String
        json_string = json.dumps(annotations)
        file.write(f"{json_string}\n")

# %%

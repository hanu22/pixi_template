# %%
from pathlib import Path

from mlops.training.scripts.convert import (
    convert_cats_to_doc,
    convert_ners_to_doc,
    convert_ners_to_spancat_doc,
)
from mlops.training.scripts.load import read_processed_file
from mlops.training.scripts.split import split_train_dev, split_train_dev_multi_label

# %%
df = read_processed_file(
    Path(
        "mlops",
        "features",
        "processed",
        "data",
        "data_keep.csv",
    )
)

df.info()

# %% Classification

# Get unique labels
labels = set([label for labels in df["label"] for label in labels])

df["doc"] = df.apply(
    lambda row: convert_cats_to_doc(
        text=row["text"],
        text_labels=row["label"],
        labels=labels,
    ),
    axis=1,
)

# For multi-class, take the first label in the list (the only one)
df["stratify"] = df["label"].apply(lambda x: x[0])

# %% NER

df["doc"] = df.apply(
    lambda row: convert_ners_to_doc(
        row_gt=row.loc[[]],
        text=row.loc["text"],
    ),
    axis="columns",
)

# Convert few entities from NERs to SpanCat
df["doc"] = df.apply(
    lambda row: convert_ners_to_spancat_doc(
        doc=row["doc"],
        span_cats=[],
    ),
    axis="columns",
)

# Take the sum of available NERs for each row
df["stratify"] = (
    df[
        [
            "",
        ]
    ]
    .map(lambda cell: 1 if "None" not in cell else 0)
    .sum(axis=1)
)

# %%
# Create spacy and JSON files

# Multi-class
df_train, df_dev = split_train_dev(
    df,
    stratify_by=df["stratify"],
)

df_train["stratify"].value_counts()
df_dev["stratify"].value_counts()

# Multi-label: use an iterative stratifier
df_train, df_dev = split_train_dev_multi_label(df)

# Check if the split is correct
one_hot_columns = list(labels)

df_train[one_hot_columns].sum().sort_values(ascending=False)
df_dev[one_hot_columns].sum().sort_values(ascending=False)

# %%

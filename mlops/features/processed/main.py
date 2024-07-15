# %%
from pathlib import Path

import pandas as pd

from mlops.features.processed.scripts.load import read_raw_file
from mlops.features.processed.scripts.preprocess import (
    coarse_gtin_to_length,
    sanitize,
)
from mlops.features.processed.scripts.similarity import filter_similar_text

# %%
df = read_raw_file(
    Path(
        "mlops",
        "features",
        "raw",
        "data",
        "data.csv",
    )
)

df.info()

# %%
# Preprocess data
df = sanitize(
    df,
    cols=["text"],
)

# Make sure gtin length is 13
df["gtin"] = df["gtin"].apply(coarse_gtin_to_length)

# Keep only rows with text length more than 3
df = df[df["text"].apply(len) > 3]

df.info()

# %%
# Remove similar texts for each market
markets = df["market"].unique()

df_markets_keep = []
df_markets_remove = []

for market in markets:
    df_market = df.query("market == @market").copy()
    print(f"{market=}, {len(df_market)=}")

    df_keep, df_remove = filter_similar_text(
        df=df_market,
        threshold=0.8,
    )

    df_markets_keep.append(df_keep)
    df_markets_remove.append(df_remove)

df_keep = pd.concat(df_markets_keep)
df_remove = pd.concat(df_markets_remove)

assert len(df_keep) + len(df_remove) == len(df)

df_remove.info()
df_keep.info()

# %%
# EDA

# %%
# Resampling
# Multi-class (each sample could only have 1 label)
# Multi-label (each sample could have more than 1 label)

# One-hot encoded labels: multi-class and multi-label
one_hot_columns = []

# Add "negative" column for rows with no labels (negative class)
df_keep["negative"] = df_keep[one_hot_columns].apply(
    lambda row: 1 if row.sum() == 0 else 0,
    axis=1,
)

one_hot_columns = one_hot_columns + ["negative"]

# Count the occurrence of each label
label_counts = df_keep[one_hot_columns].sum().sort_values(ascending=False)

# Get a list of columns with less than 100 occurrences to drop
one_hot_columns_drop = label_counts[label_counts < 100].index.to_list()

df_keep = df_keep.drop(
    labels=one_hot_columns_drop,
    axis=1,
)

# Use remaining one-hot columns to generate labels
one_hot_columns_remaining = [
    one_hot_column
    for one_hot_column in one_hot_columns
    if one_hot_column not in one_hot_columns_drop
]

df_keep["label"] = df_keep[one_hot_columns_remaining].apply(
    lambda row: row.index[row == 1].to_list(),
    axis=1,
)

# Normal labels: multi-class and multi-label

df_keep["label"].value_counts()

# Drop labels with less than 100 occurrences
df_keep = df_keep.groupby("label").filter(lambda x: len(x) >= 100)

df_keep["label"].value_counts()

# Create a set of labels for each example to remove duplicates
# Convert to list for Spacy
df_keep = df_keep.groupby("text")["label"].apply(set).apply(list).reset_index()

# For multiclass, the list of labels should only have 1 label
# Resolve/drop rows with more than 1 label
len(df_keep[df_keep["label"].apply(len) > 1])
df_keep = df_keep[df_keep["label"].apply(len) == 1]

# %%
df_keep.to_csv(
    Path(
        "mlops",
        "features",
        "processed",
        "data",
        "data_keep.csv",
    ),
    index=True,
)

df_remove.to_csv(
    Path(
        "mlops",
        "features",
        "processed",
        "data",
        "data_remove.csv",
    ),
    index=True,
)

# %%

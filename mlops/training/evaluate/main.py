# %%
from pathlib import Path

import pandas as pd
import spacy

from mlops.training.scripts.convert import convert_cats_to_doc
from mlops.training.scripts.eda import plot_cats_per_market
from mlops.training.scripts.evaluate import (
    calculate_cats_accuracy,
    predict_cats,
)
from mlops.training.scripts.load import read_processed_file

# %%
# Load data
df = read_processed_file(
    Path(
        "features",
        "processed",
        "data",
        "data.csv",
    )
)

df_dev = pd.read_json(
    Path(
        "mlops",
        "training",
        "assets",
        "dev.json",
    )
)

df_train = pd.read_json(
    Path(
        "mlops",
        "training",
        "assets",
        "train.json",
    )
)

# %%
# Load trained model (downloaded from AML)
nlp = spacy.load(
    Path(
        "mlops",
        "model",
    )
)

# %%
# Predict classes
# Subset original dataframe with testing dataframe
# to have access to other columns
df_test = df.loc[df_dev.index]


df_test["doc"] = df_test.apply(
    lambda s: convert_cats_to_doc(
        text=s["text"],
        text_labels=s["label"],
        labels=df["label"].unique(),
    ),
    axis=1,
)

# Predict classes using trained model
df_test["prediction"] = df_test["text"].apply(
    lambda x: predict_cats(
        x,
        nlp,
    )
)

df_test["evaluation"] = df_test.apply(
    lambda x: calculate_cats_accuracy(
        labels_gt=x["doc"].cats,
        labels_predicted=x["prediction"],
    ),
    axis=1,
)

# %%
# EDA
# Filter rows with errors
errors = df_test[df_test["evaluation"] == 0]

# Show the errors
plot_cats_per_market(
    df=errors,
    save_path=Path(
        "mlops",
        "training",
        "eda",
        "errors_per_market",
    ),
)

# Show the accuracies
df_test.groupby(
    [
        "label",
        "evaluation",
        "text",
    ],
    as_index=False,
).mean().groupby("label")["evaluation"].mean()

# %%

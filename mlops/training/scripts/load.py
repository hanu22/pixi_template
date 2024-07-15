import ast
from pathlib import Path

import pandas as pd


def read_processed_file(file_path: Path) -> pd.DataFrame:
    """Reads file as dataframe.

    Convert string list objects into literal list object

    Args:
        file_path (Path): file path

    Returns:
        pd.DataFrame: dataframe
    """
    df = (
        pd.read_csv(
            file_path,
            usecols=[
                "text",
                "label",
            ],
            dtype={
                "label": str,
            },
        )
        .assign(label=lambda df: df["label"].str.upper())
        .assign(label=lambda x: x["label"].apply(ast.literal_eval))
        .set_index("pvid")
        .dropna()
    )

    return df

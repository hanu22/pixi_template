from functools import lru_cache
from pathlib import Path

import pandas as pd


@lru_cache
def read_raw_file(file_path: Path) -> pd.DataFrame:
    """Reads file as dataframe.

    Args:
        file_path (Path): file path

    Returns:
        pd.DataFrame: dataframe
    """
    df = (
        pd.read_csv(
            file_path,
            dtype=str,
            na_values=["", " ", "None"],
        )
        .dropna(subset=["text"])
        .drop_duplicates(subset=["text"])
        .fillna("None")
        .set_index("pvid")
    )

    return df

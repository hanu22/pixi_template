import re
from unicodedata import normalize

import ftfy
import numpy as np
import pandas as pd


def coarse_gtin_to_length(
    gtin: str,
    length: int = 13,
) -> str:
    """Coarse gtin to length.

    GTIN length in the database is not always 13 digits
    The product library will usually fix this by padding or stripping a GTIN to 13

    Args:
        gtin (str): gtin
        length (int, optional): required gtin length. Defaults to 13.

    Returns:
        str: _description_
    """
    # Pad with leading zeros to length
    if len(gtin) < length:
        gtin = gtin.zfill(13)

    # Get only last length digits
    elif len(gtin) > length:
        gtin = gtin[-length:]

    assert len(gtin) == 13, f"Warning, could not coarse to {length} digits"

    return gtin


def sanitize(
    df: pd.DataFrame,
    cols: list[str],
) -> pd.DataFrame:
    """Sanitizes data.

    Normalize decimal points
    Normalize unicode
    Replace empty string "" with nan
    Remove leading, trailing, or 2+ spaces

    Drop columns rows with any missing values
    Drop columns with duplicated values

    Args:
        df (pd.DataFrame): dataframe
        cols (list[str]) : list of columns to sanitize

    Returns:
        pd.DataFrame: dataframe
    """
    df[cols] = (
        df[cols]
        .map(normalise_decimal_point)
        .map(normalize_unicode)
        .map(str.strip)
        .replace(
            to_replace=r"\s+",
            value=" ",
            regex=False,
        )
        .replace(
            to_replace="",
            value=np.nan,
            regex=False,
        )
    )

    # Convert float columns back to str
    df = (
        df.dropna(subset=cols)
        .drop_duplicates(subset=cols)
        .astype({col: str for col in cols})
    )

    return df


def normalise_decimal_point(text: str) -> str:
    """Normalises decimal points between digits.

    Change text like '18,1' to '18.1'
    This occurs in some EU markets

    Args:
        text (str): text

    Returns:
        str: normalised text
    """
    pattern = r"""
        (\d+)  # One or more digits
        (\,)   # Comma
        ([\d]+)  # One or more digits
        """

    # Replace comma with decimal point
    text = re.sub(
        pattern,
        lambda ele: ele[1] + "." + ele[3],
        text,
        flags=re.VERBOSE,
    )

    return text


def normalize_unicode(text: str) -> str:
    """Normalizes unicode.

    Args:
        text (str): text

    Returns:
        str: normalized text
    """
    # Fix broken Unicode and mojibake (encoding mix-ups)
    text = ftfy.fix_text(text)

    # Compatibility Decomposition, followed by Canonical Composition
    text = normalize("NFKC", text)

    return text

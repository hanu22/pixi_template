import re
from unicodedata import normalize

import ftfy


def sanitize(text: str) -> str:
    """Sanitizes text.

    Remove leading, trailing, or 2+ spaces

    Drop columns rows with any missing values
    Drop columns with duplicated values

    Args:
        text (str): text to sanitize

    Returns:
        pd.DataFrame: dataframe
    """
    text = normalise_decimal_point(text)
    text = normalize_unicode(text)
    text = text.strip()

    # replace 2+ spaces with 1 space
    text = re.sub(r"\s+", " ", text)

    return text


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

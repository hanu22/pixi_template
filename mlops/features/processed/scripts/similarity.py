from collections import Counter

import pandas as pd
from datasketch import MinHash, MinHashLSH


def get_indices_to_keep(
    indices: list[list[str]],
) -> tuple[list[str], list[str]]:
    """Gets a list of indices to keep.

    Args:
        indices (list[list[str]]): indices to process

    Assuming we have 3 texts which are similar.
    Their indices are: 3, 12, 9.

    Each of them will have a list of similar text indices such as:
    Text with index 3 has these similar texts indices: [3, 12 , 9]
    Text with index 12 has these similar texts indices: [12, 3, 9]
    Text with index 9 has these similar texts indices: [9, 3, 12]

    The goal is to keep only 1 index.

    Step 1 > make the 3 lists of indices unique > [3 ,9 , 12]
    Step 2 > Sample 2 indices randomly to be removed > [9, 12]

    Returns:
        tuple[list[str], list[str]]: indices to keep and removed
    """
    # Apply set() to only keep unique tuples of indices
    # convert from list of lists to list of tuples
    # as set() only works on hashable types
    indices_unique = set(map(tuple, indices))

    indices_keep = []

    # Count the frequency of each index to keep the most frequent
    indices_unique_flat = [index for tup in indices_unique for index in tup]
    frequency = Counter(indices_unique_flat)

    # Get the frequencies of the current indices tuple
    for indices_unique_tup in indices_unique:
        tup_frequency = {}

        for index in indices_unique_tup:
            tup_frequency[index] = frequency[index]

        # Get the index with the highest frequency to keep
        index_max = max(tup_frequency, key=tup_frequency.get)
        indices_keep.append(index_max)

    # Remove duplicated indices
    # Do not sort to keep similar indices together
    indices_keep_unique = list(set(indices_keep))
    indices_remove_unique = list(set(indices_unique_flat) - set(indices_keep_unique))

    return indices_keep_unique, indices_remove_unique


def filter_similar_text(
    df: pd.DataFrame,
    threshold: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Filters similar values using MinHash LSH.

    TODO: try Asynchronous?

    Drops these values from the input dataframe
    Expects df to have a column called 'text'
    Returns the original input dataframe for accuracy evaluation

    Args:
        df (pd.DataFrame): Dataframe with similar values
        threshold (int): threshold to match on

    Returns:
        pd.DataFrame: Dataframe without similar values
        pd.DataFrame: Input dataframe
    """
    # Split each text on spaces >
    # Remove duplicates using set >
    # Encode each token and return a list
    df["tokens"] = (
        df["text"]
        .apply(str.split)
        .apply(set)
        .apply(lambda tokens: [token.encode("utf8") for token in tokens])
    )

    # Create MinHash permutations for each set of tokens
    df["mhs"] = [
        MinHash(
            num_perm=128,
            seed=42,
        )
        for _ in range(len(df))
    ]

    # Update each mhs with tokens
    df[["tokens", "mhs"]].apply(
        lambda row: row["mhs"].update_batch(row["tokens"]),
        axis=1,
    )

    # Create LSH index
    lsh = MinHashLSH(
        threshold=threshold,
        num_perm=128,
    )

    # Insert all mhs into LSH index
    # Each mhs key in the LSH index is it's index (0, 1, 2, etc.)
    # Use ind as keys of the LSH index
    for ind, mh in enumerate(df["mhs"]):
        lsh.insert(f"{ind}", mh)

    # Save each query similar keys
    # This includes the query mh index itself
    df["mhs_keys"] = df["mhs"].apply(lsh.query)

    # Get indices to keep and remove
    indices_keep, indices_remove = get_indices_to_keep(df["mhs_keys"])

    # Convert indices to int
    indices_keep = list(map(int, indices_keep))
    indices_remove = list(map(int, indices_remove))

    df = df.drop(columns=["tokens", "mhs", "mhs_keys"])

    df_keep = df.iloc[indices_keep]
    df_remove = df.iloc[indices_remove]

    return df_keep, df_remove

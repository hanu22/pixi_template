from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import missingno as msno
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud

sns.set_style("white")
sns.set_palette("Set2")


# TODO: can we plot NERs distinctiveness?


def plot_samples_per_market(
    df: pd.DataFrame,
    n_markets: int,
):
    """Plots number of samples per market.

    Args:
        df (pd.DataFrame): dataframe
        n_markets (int): number of markets to plot
    """
    fig, ax = plt.subplots(
        figsize=(6, 6),
        tight_layout=True,
    )

    # Number of samples per market for the top tm_no
    df["market"].value_counts().head(n_markets).plot(kind="bar", ax=ax)

    ax.set(
        title="Number of samples per market",
        xlabel="Market",
        ylabel="Samples No.",
    )

    # Save figure
    fig.savefig(
        Path(
            "mlops",
            "features",
            "processed",
            "eda",
            "samples_per_market.png",
        )
    )


def plot_samples_per_market_hue(
    df: pd.DataFrame,
    n_markets: int,
    hue: str,
):
    """Plots number of samples per market.

    Args:
        df (pd.DataFrame): dataframe
        n_markets (int): number of TM to plot
        hue (str): column for hue
    """
    fig, ax = plt.subplots(
        figsize=(6, 6),
        tight_layout=True,
    )

    # Number of samples per market for the top 10 per category
    sns.countplot(
        x="market",
        hue=hue,
        data=df,
        order=df["market"].value_counts().head(n_markets).index.to_list(),
        ax=ax,
    )

    ax.set(
        title="Number of samples per market",
        xlabel="Market",
        ylabel="Samples No.",
    )

    # Save figure
    fig.savefig(
        Path(
            "mlops",
            "features",
            "processed",
            "eda",
            "samples_per_market_hue.png",
        )
    )


def __plot_missing_values(
    df: pd.DataFrame,
    category: str,
    save_as: str,
):
    """Plots missing value for each column in dataframe.

    Args:
        df (pd.DataFrame): dataframe
        category (str): category name
        save_as (str): name of figure
    """
    fig, ax = plt.subplots(tight_layout=True)

    # Create bar plot of missing values across all columns
    msno.bar(
        df,
        fontsize=12,
        ax=ax,
    )

    ax.set_title(f"Number of samples for level 1 category: {category}")
    ax.set_ylabel("")

    fig.savefig(
        Path(
            "mlops",
            "features",
            "processed",
            "eda",
            save_as,
        )
    )


def plot_missing_values(
    df: pd.DataFrame,
    per_category: bool = False,
):
    """Plots missing value for each column.

    Args:
        df (pd.DataFrame): dataframe
        per_category (bool, optional): if true plots per category level one.
        Defaults to False.
    """
    if per_category:
        categories = df["cat_l1"].unique()

        for category in categories:
            __plot_missing_values(
                df[df["cat_l1"] == category],
                category=category.title(),
                save_as=f"missing_values_{category}_category.png",
            )
    else:
        __plot_missing_values(
            df,
            category="all",
            save_as="missing_values_all_categories.png",
        )


def plot_word_cloud(
    words: list,
    save_as: str,
):
    """Plots word cloud.

    Args:
        words (list): list of words
        save_as (str): name of figure
    """
    # Join word list
    text = " ".join(words.dropna())

    wc = WordCloud(collocations=False).generate(text)

    fig, ax = plt.subplots(tight_layout=True)
    plt.imshow(wc, interpolation="bilinear")
    ax.axis("off")

    fig.savefig(
        Path(
            "mlops",
            "features",
            "processed",
            "eda",
            save_as,
        )
    )


def plot_text_length_hist(
    s: pd.Series,
    by: Literal["words", "characters"],
):
    """Plots words or characters counts histogram for a column.

    Does not include rows with "None"

    Args:
        s (pd.Series): column to plot
        by (Literal["words", "characters"]): options
    """
    if by == "characters":
        # Get length of text including spaces
        s_count = s[~(s == "None")].apply(len)

    elif by == "words":
        # Split on " " and get length of list of words
        s_count = s[~(s == "None")].apply(
            lambda x: len(
                x.split(" "),
            )
        )

    fig, ax = plt.subplots(tight_layout=True)

    sns.histplot(
        s_count,
        # bins=np.arange(s_count.min(), 510, 10),
        ax=ax,
    )

    ax.set(
        xlabel="Count",
        ylabel="Frequency",
        title=f"{by.title()} count distribution for {s.name}",
    )

    fig.savefig(
        Path(
            "mlops",
            "features",
            "processed",
            "eda",
            f"{by.title()}_count_{s.name}.png",
        )
    )


def get_tf(s: pd.Series) -> pd.DataFrame:
    """Gets uni-gram term frequency for a column.

    Args:
        s (pd.Series): Series containing text

    Returns:
        pd.DataFrame: corpus with frequencies
    """
    vectorizer = CountVectorizer(analyzer="word")

    # Get the bag of words matrix
    sparse_matrix = vectorizer.fit_transform(s)

    # Each row is a BOW for one word -> sum to get count
    corpus_frequencies = sparse_matrix.toarray().sum(axis=0)

    # Get the corpus as an array of words
    corpus = vectorizer.get_feature_names_out()

    df_tf = (
        pd.DataFrame(
            data={
                "corpus": corpus,
                "frequency": corpus_frequencies,
            }
        )
        .sort_values(
            axis="index",
            by=["frequency", "corpus"],
            ascending=False,
        )
        .reset_index(drop=True)
    )

    return df_tf


def get_tf_per_unique_values(
    df: pd.DataFrame,
    col_name: str,
) -> pd.DataFrame:
    """Gets uni-gram term frequency for a column per unique values in another column.

    e.g. unique values in market

    Args:
        df (pd.DataFrame): Dataframe with column of grouping
        col_name (str): name of grouping column in df

    Returns:
        pd.DataFrame: corpus with frequencies. Each line is for one unique value
    """
    # TODO: needs improvements
    df_tf_all = df.groupby(col_name).agg({"text": get_tf})

    return df_tf_all


def count_ners(s: pd.Series) -> int:
    """Counts entities in a column of lists.

    Some cells have ["None"] > replace with np.nan > drop

    Args:
        s (pd.Series): column with entities

    Returns:
        int: sum of entities
    """
    return (
        s.map(lambda cell: np.nan if cell == ["None"] else cell)
        .dropna()
        .apply(len)
        .sum()
    )


def plot_ner_count(
    df: pd.DataFrame,
    cols: list[str],
):
    """Plots count of ground truth entities, per type.

    Args:
        df (pd.DataFrame): dataframe with ground truth entities
        cols (list[str]): list of NER columns
    """
    # Group by market ->
    # count NERs per column per market ->
    # sum NERs per column ->
    # pd.Series
    s = df.groupby("market").agg(count_ners).sum()

    # Select only NER rows in series
    s = s.loc[cols].sort_values(ascending=False)

    fig, ax = plt.subplots(
        tight_layout=True,
    )

    # Count of all entities
    sns.barplot(
        x=s.index,
        y=s.values,
        ax=ax,
    )

    ax.set(
        title="Count of entities per type",
        xlabel="Samples No.",
        ylabel="",
    )

    ax.tick_params(
        axis="x",
        rotation=45,
    )

    # Save figure
    fig.savefig(
        Path(
            "mlops",
            "features",
            "processed",
            "eda",
            "ents_per_type.png",
        )
    )


def plot_ner_count_per_market(
    df: pd.DataFrame,
    cols: list[str],
):
    """Plots count of ground truth entities, per market, per type.

    Args:
        df (pd.DataFrame): dataframe with ground truth entities
        cols (list[str]): list of NER columns
    """
    # Group by market ->
    # count NERs per column per market ->
    # sum NERs per column ->
    # pd.Series
    s = df.groupby("market").agg(count_ners)

    # Select only NER rows in series
    s = s[cols]

    # Convert data to long format
    s = (
        s.melt(ignore_index=False)
        .reset_index()
        .sort_values(
            by=["market", "value"],
            ascending=False,
        )
    )

    fig, ax = plt.subplots(
        tight_layout=True,
    )

    sns.barplot(
        x="value",
        y="market",
        hue="variable",
        data=s,
        ax=ax,
    )

    ax.set(
        title="Count of entities per market and type",
        xlabel="Samples No.",
        ylabel="",
    )

    fig.savefig(
        Path(
            "mlops",
            "features",
            "processed",
            "eda",
            "ents_per_type_per_market.png",
        )
    )


def plot_multi_labels_counts(
    df: pd.DataFrame,
    cols: list[str],
):
    """Plots multi-labels count in one-hot encoding format.

    Args:
        df (pd.DataFrame): dataframe
        cols (list[str]): list of columns with one hot encoding labels
    """
    # Count the occurrence of each label
    label_counts = df[cols].sum().sort_values(ascending=False)

    fig, ax = plt.subplots(
        figsize=(6, 6),
        tight_layout=True,
    )

    label_counts.plot(
        kind="bar",
        figsize=(10, 10),
        title="Label Counts",
        ax=ax,
    )

    fig.savefig(
        Path(
            "mlops",
            "features",
            "processed",
            "eda",
            "labels_count.png",
        )
    )


def plot_label_count(df: pd.DataFrame):
    """Plots value count distribution for labels.

    Args:
        df (pd.DataFrame): Pandas Dataframe with labels
    """
    fig, ax = plt.subplots(
        figsize=(6, 6),
        tight_layout=True,
    )

    sns.histplot(
        data=df,
        x="label",
        ax=ax,
    )

    ax.set(
        title="Number of Labels",
        xlabel="Labels",
        ylabel="Samples No.",
    )

    ax.tick_params(
        axis="x",
        rotation=90,
    )

    fig.savefig(
        Path(
            "mlops",
            "features",
            "processed",
            "eda",
            "label_count.png",
        )
    )


def count_unique_occurrences(column: pd.Series) -> pd.DataFrame:
    """Gets the count of unique strings in a column of lists of strings.

    Args:
        column (pd.Series): column of lists of strings

    Returns:
        pd.DataFrame: dataframe of counts
    """
    # Flatten the column into a single list
    flat_list = [item for sublist in column for item in sublist]

    # Create an empty dictionary to store the counts
    unique_counts = {}

    for item in flat_list:
        if item in unique_counts:
            # Increment the count if the item is already in the dictionary
            unique_counts[item] += 1

        else:
            # Add the item to the dictionary with a count of 1 if it's not already there
            unique_counts[item] = 1

    df = pd.DataFrame.from_dict(
        unique_counts,
        orient="index",
        columns=["count"],
    )

    return df

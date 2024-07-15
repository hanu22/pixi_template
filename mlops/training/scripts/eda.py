import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid")


def plot_metrics(
    metrics_path: Path,
    title: str,
) -> pd.DataFrame:
    """Plots metrics graph and table.

    Args:
        metrics_path (Path): path to metrics file
        title (str): title

    Returns:
        pd.DataFrame: metric dataframe
    """
    with open(metrics_path) as json_data:
        metrics = json.load(json_data)

    # Turn column of dictionary of metrics per ENT type into df
    df_raw = pd.DataFrame(metrics["performance"]["ents_per_type"])

    # Convert from wide to long format
    df = (
        pd.melt(df_raw, ignore_index=False)
        .reset_index()
        .rename(
            columns={
                "index": "measure",
                "variable": "ENT",
            }
        )
    )

    df["value"] = df["value"] * 100
    df = df.sort_values(by="ENT")

    g = sns.catplot(
        x="measure",
        y="value",
        hue="ENT",
        data=df,
        kind="bar",
        palette="dark",
        alpha=0.6,
        height=6,
    )

    g.set_axis_labels("", "Percentage")

    g.set(
        ylim=(0, 100),
        yticks=range(0, 110, 10),
        title=(title),
    )

    g.despine(left=True)

    ax = g.facet_axis(0, 0)

    # Iterate through the axes containers
    for c in ax.containers:
        labels = [f"{round((v.get_height()))}" for v in c]
        ax.bar_label(
            c,
            labels=labels,
            label_type="edge",
        )

    return df


def plot_ner_per_market(df: pd.DataFrame):
    """Plots ner average percentage of overlap, per market, per type.

    Args:
        df (pd.DataFrame): Dataframe with ground truth data
    """
    # Convert data to long format
    df = (
        df.groupby("market")
        .mean()
        .melt(ignore_index=False)
        .reset_index()
        .sort_values(by="value")
    )

    # Entities per market
    f, ax = plt.subplots(
        figsize=(
            12,
            30,
        )
    )

    sns.set_color_codes("pastel")
    sns.barplot(
        x="value",
        y="market",
        hue="variable",
        data=df,
        ci=None,
    ).set(title="Accuracy per Market per Entity")

    ax.legend(
        ncol=1,
        loc="lower right",
        frameon=True,
    )


def plot_cats_per_market():
    pass

from pathlib import Path

import pandas as pd

df_markets = pd.read_csv(
    Path(
        "mlops",
        "features",
        "raw",
        "data",
        "markets.csv",
    )
)

markets_english = df_markets.loc[
    df_markets["language_code"].str.contains("en"), "market_code"
].to_list()

markets_non_english = df_markets.loc[
    ~df_markets["language_code"].str.contains("en"), "market_code"
].to_list()

markets = markets_english + markets_non_english


def map_markets(df: pd.DataFrame) -> pd.DataFrame:
    """Maps markets to existing codes.

    Args:
        df (pd.DataFrame): dataframe

    Returns:
        pd.DataFrame: filtered dataframe
    """
    df["market"] = df["market"].replace(
        {
            "BE,LU": "BE",
            "NL, LU": "NL",
            "NL,BE": "NL",
            "CZ,US": "US",
            "GB,IE,NL": "GB,IE",
            "GB,IE,DE,DK": "GB,IE",
            "GB": "GB,IE",
            "IE": "GB,IE",
        }
    )

    return df


def get_market_id(market: str) -> str:
    """Gets market id.

    Args:
        market (str): market code

    Returns:
        str: market id
    """
    market_id = df_markets.loc[
        df_markets["market_code"] == market,
        "market_id",
    ].values[0]

    # params expects a string
    return str(market_id)

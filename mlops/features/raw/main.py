# %%
from pathlib import Path

import pandas as pd

from mlops.features.raw.scripts.extract import run
from mlops.features.raw.scripts.load import read_raw_file
from mlops.features.raw.scripts.preprocess import (
    get_market_id,
    map_markets,
    markets,
    markets_english,
    markets_non_english,
)
from mlops.features.raw.scripts.translator import translate_market

# %%
from_date = "2021-06-01"
to_date = "2024-06-01"

for market in markets:
    # Query expects a market id not a code
    market_id = get_market_id(market)
    print(f"{market=}, {market_id=}")

    df = run(
        path=Path(
            "mlops",
            "features",
            "raw",
            "data",
            "query.sql",
        ),
        params={
            "from_date": from_date,
            "to_date": to_date,
            "market_id": market_id,
        },
    )

    df.to_csv(
        Path(
            "mlops",
            "features",
            "raw",
            "data",
            f"{market}.csv",
        ),
        index=False,
    )

# %%
for market in markets_non_english:
    df = read_raw_file(
        Path(
            "mlops",
            "features",
            "raw",
            "data",
            f"{market}.csv",
        ),
    )

    print(f"{market=}, {len(df)=}")

    # Statements were concatenated with " | ", replace with a full stop
    df["text"] = df["text"].str.replace(" | ", ". ", regex=False)

    # Original text will be copied to 'text_orig' column
    df = translate_market(market, df)

    df.to_csv(
        Path(
            "mlops",
            "features",
            "raw",
            "data",
            f"{market}_translated.csv",
        ),
        index=True,
    )

# %%
dfs = []

for market in markets:
    print(f"{market=}")

    if market in markets_non_english:
        file_name = f"{market}_translated.csv"

    elif market in markets_english:
        file_name = f"{market}.csv"

    df = read_raw_file(
        Path(
            "mlops",
            "features",
            "raw",
            "data",
            file_name,
        ),
    )

    if market in markets_english:
        # Was only done for non-english markets
        df["text_orig"] = df["text"]
        df["text"] = df["text"].str.replace(" | ", ". ", regex=False)

    df = map_markets(df)

    dfs.append(df)

df = pd.concat(dfs)

df.to_csv(
    Path(
        "mlops",
        "features",
        "raw",
        "data",
        "data.csv",
    ),
    index=True,
)

# %%

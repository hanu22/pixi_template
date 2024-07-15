import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv(".env")

SQL_DEV_USERNAME = os.getenv("SQL_DEV_USERNAME")
SQL_DEV_PASSWORD = os.getenv("SQL_DEV_PASSWORD")

host = "sql-dev-vm1.development.local"
database = "MergedILabelServer_Daily"
driver = "SQL Server"


engine = create_engine(
    f"mssql+pyodbc://"
    f"{SQL_DEV_USERNAME}:"
    f"{SQL_DEV_PASSWORD}@"
    f"{host}/"
    f"{database}?"
    f"driver={driver}"
)


def run(
    path: Path,
    params: dict = None,
) -> pd.DataFrame:
    """Runs a query with parameters.

    Args:
        path (Path): full query path
        params (dict, optional): query parameters. Defaults to None.

    Returns:
        pd.DataFrame: query output
    """
    with engine.connect() as con:
        with path.open("r") as f:
            df = pd.read_sql(
                sql=text(f.read()),
                con=con,
                params=params,
            ).astype(str)

    return df

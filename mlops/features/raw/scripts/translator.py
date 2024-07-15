import argparse
import asyncio
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Union

import pandas as pd
import requests
from dotenv import load_dotenv

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%d/%m/%y %H:%M:%S",
)

load_dotenv(".env")

TRANSLATOR_API_KEY = os.getenv("TRANSLATOR_API_KEY_DEV")
endpoint = r"https://api.cognitive.microsofttranslator.com"

headers = {
    "Ocp-Apim-Subscription-Key": TRANSLATOR_API_KEY,
    "Ocp-Apim-Subscription-Region": "northeurope",
    "Content-type": "application/json",
    "X-ClientTraceId": str(uuid.uuid4()),
}

# Map existing language codes to expected Azure Translator language codes
# FIXME: 'BE' has multiple languages, french and dutch
# IT's best to use a product's language_code instead of market_code
translator_language_codes = {
    "BE": "nl",
    "CZ": "cs",
    "HU": "hu",
    "IT": "it",
    "NL": "nl",
    "PL": "pl",
    "RO": "ro",
    "SK": "sk",
}


def translate(
    texts: list[str],
    timeout: float,
    from_language: str = None,
    to_language: str = "en",
) -> Union[list[dict], None]:
    """Translates texts using Azure Translator API.

    Args:
        texts (list): a list of texts to translate
        timeout (float): request timeout
        from_language (str): language code to translate from. Defaults to None.
        to_language (str): language code to translate to. Defaults to "en".

    Returns:
        Union[list[dict], None]: list of translation dictionaries:

        {
            "translations": [
                {
                    "text": str,
                    "to": str,
                }
            ],
        }

        or None if an error is raised
    """
    # API expects a list of dictionaries, one for each text
    body = [{"text": text} for text in texts]

    logging.info("Posting translation request...")

    params = {
        "api-version": "3.0",
        "from": from_language,
        "to": to_language,
    }

    try:
        response = requests.post(
            endpoint + "/translate",
            params=params,
            headers=headers,
            json=body,
            timeout=timeout,
        )

    except requests.exceptions.Timeout:
        logging.error(f"TimeoutError after {timeout}")
        return None

    else:
        try:
            response.raise_for_status()

        except requests.exceptions.HTTPError as e:
            logging.error(f"HTTPError: {e}")
            return None

        else:
            response = response.json()
            return response


def translate_market(
    market: str,
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Translates a market texts into English.

    Expects a 'text' column
    Translation results are saved into new columns

    Args:
        market (str): market code
        df (pd.DataFrame): dataframe

    Returns:
        pd.DataFrame: dataframe with translated text
    """
    # Each translate request is limited to 50,000 characters, including translation
    # Split the text column into chunks of 200
    n = 200
    df_chunks = [df[i : i + n] for i in range(0, len(df), n)]

    text_translated_chunks = []

    for df_chunk in df_chunks:
        response = translate(
            texts=df_chunk["text"].tolist(),
            timeout=20,
            from_language=translator_language_codes[market],
            to_language="en",
        )

        assert len(response) == len(df_chunk), "Missing translation responses"
        text_translated_chunk = [r["translations"][0]["text"] for r in response]
        text_translated_chunks.extend(text_translated_chunk)

        # Sleep to avoid rate limiting
        time.sleep(2)

    # Copy text to a new column before overwriting for reference
    df["text_orig"] = df["text"]
    df["text"] = text_translated_chunks

    return df


async def translate_async(
    texts: list[str],
    timeout: float,
    from_language: str = None,
    to_language: str = "en",
) -> Union[list[dict], None]:
    """Translates texts using Azure Translator API.

    Args:
        texts (list): a list of texts to translate
        timeout (float): request timeout
        from_language (str): language code to translate from. Defaults to None.
        to_language (str): language code to translate to. Defaults to "en".

    Returns:
        Union[list[dict], None]: list of translation dictionaries:

        {
            "translations": [
                {
                    "text": str,
                    "to": str,
                }
            ],
        }

        or None if an error is raised
    """
    # API expects a list of dictionaries, one for each text
    body = [{"text": text} for text in texts]

    logging.info("Posting translation request...")

    params = {
        "api-version": "3.0",
        "from": from_language,
        "to": to_language,
    }

    try:
        response = requests.post(
            endpoint + "/translate",
            params=params,
            headers=headers,
            json=body,
            timeout=timeout,
        )

    except requests.exceptions.Timeout:
        logging.error(f"TimeoutError after {timeout}")
        return None

    else:
        try:
            response.raise_for_status()

        except requests.exceptions.HTTPError as e:
            logging.error(f"HTTPError: {e}")
            return None

        else:
            response = response.json()
            return response


async def translate_market_async(
    market: str,
    df: pd.DataFrame,
):
    """Translates a market texts into English.

    Expects a 'text' column
    Translation results are saved into new columns

    Args:
        market (str): market code
        df (pd.DataFrame): dataframe

    Returns:
        pd.DataFrame: dataframe with translated text
    """
    logging.info("Translating market concurrently...")

    # Each translate request is limited to 50,000 characters, including translation
    # Split the text column into chunks of 200
    n = 20
    df_chunks = [df[i : i + n] for i in range(0, len(df), n)]

    tasks = set()

    for df_chunk in df_chunks:
        task = asyncio.create_task(
            translate_async(
                texts=df_chunk["text"].tolist(),
                timeout=20,
                from_language=translator_language_codes[market],
                to_language="en",
            )
        )

        tasks.add(task)

        # To prevent keeping references to finished tasks forever
        task.add_done_callback(tasks.discard)

        # Sleep to avoid rate limiting
        time.sleep(1)

    results = await asyncio.gather(
        *tasks,
        return_exceptions=True,
    )

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-m",
        "--market",
        type=str,
        required=True,
    )

    parser.add_argument(
        "-dp",
        "--data_path",
        type=str,
        required=True,
    )

    args = parser.parse_args()

    market = args.market
    data_path = Path(args.data_path)

    df = (
        pd.read_csv(
            Path(
                data_path,
                f"{market}.csv",
            ),
            na_values=["", " ", "None"],
        )
        .dropna(subset=["text"])
        .drop_duplicates(subset=["text"])
        .assign(text_orig=lambda df: df["text"])
    )

    print(f"{market=}, {len(df)=}")

    # Statements were concatenated with |, replace with a full stop
    df["text"] = df["text"].str.replace(" |", ". ")

    result_chunks = asyncio.run(
        translate_market_async(
            market,
            df,
        )
    )

    texts_translations = []

    # Loop over chunks of results > text in each chunk
    for result_chunk in result_chunks:
        for result in result_chunk:
            text_translated = result["translations"][0]["text"]
            texts_translations.append(text_translated)

    logging.info(f"{market=}, {len(texts_translations)=}")
    assert len(df) == len(texts_translations), "Missing translation responses"

    # Original text is already available in 'text_orig' column
    df["text"] = texts_translations

    df.to_csv(
        Path(
            data_path,
            f"{market}_translated.csv",
        ),
        index=False,
    )

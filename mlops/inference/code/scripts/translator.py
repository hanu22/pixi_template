import logging
import os
import uuid
from typing import Union

import requests

TRANSLATOR_API_KEY = os.getenv("TRANSLATOR_API_KEY")
endpoint = r"https://api.cognitive.microsofttranslator.com"
location = "northeurope"

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%d/%m/%y %H:%M:%S",
)

headers = {
    "Ocp-Apim-Subscription-Key": TRANSLATOR_API_KEY,
    "Ocp-Apim-Subscription-Region": location,
    "Content-type": "application/json",
    "X-ClientTraceId": str(uuid.uuid4()),
}


def translate(
    texts: list[str],
    timeout: float,
    from_language: str,
    to_language: str = "en",
) -> Union[list[dict], None]:
    """Posts a request to Azure Translator API.

    Args:
        texts (list): a list of texts to translate
        timeout (float): request timeout
        from_language (str): language to translate from
        to_language (str): language to translate to

    Returns:
        Union[list[dict], None]: list of translation dictionaries:

        {
            "detectedLanguage": {
                "language": str,
                "score": float,
            },
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
            logging.info("Returning translation dictionary...")
            return response

import json
import logging
import os
from pathlib import Path

import spacy
from azureml.contrib.services.aml_request import AMLRequest, rawhttp
from azureml.contrib.services.aml_response import AMLResponse
from pydantic import ValidationError
from scripts.preprocess import sanitize
from scripts.translator import translate
from validate import RequestModel, ResponseModel

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%d/%m/%y %H:%M:%S",
)

spacy.require_gpu()

MODEL_NAME = os.getenv("MODEL_NAME")
MODEL_VERSION = os.getenv("MODEL_VERSION")

ENVIRONMENT_NAME = os.getenv("ENVIRONMENT_NAME")
ENVIRONMENT_VERSION = os.getenv("ENVIRONMENT_VERSION")

AZUREML_MODEL_DIR = os.getenv("AZUREML_MODEL_DIR")


def create_error_response(
    code: int,
    message: str,
) -> AMLResponse:
    """Creates an error response.

    AML returns a 424 for any non-200 response in container

    Args:
            code (int): error code
            message (str): error message

    Returns:
            AMLResponse: response
    """
    body = {
        "errorType": "Error",
        "code": code,
        "message": message,
    }

    logging.error(f"Request validation error {body}.")

    response = AMLResponse(
        message=json.dumps(
            body,
            ensure_ascii=False,
        ),
        status_code=424,
    )

    response.mimetype = "application/json"

    return response


def init():
    global nlp
    global ents_target
    global ents_target_mapping

    logging.info("Loading model...")

    nlp = spacy.load(
        Path(
            AZUREML_MODEL_DIR,
            "model-best",
        ),
    )

    # Trained entities to recognize
    ents_target = []

    # Map trained entities to expected BOX-R entities
    ents_target_mapping = {}


@rawhttp
def run(request: AMLRequest):
    logging.info("HTTP trigger function processed a request.")

    try:
        # data = request  # Test locally
        data = request.get_json()

    except ValueError as error:
        return create_error_response(
            code=0,
            message=error.errors(),
        )

    else:
        try:
            RequestModel(**data)

        except ValidationError as error:
            # Only keep these keys
            keys = ["type", "loc", "msg", "input"]
            errors = [{key: e[key] for key in keys} for e in error.errors()]

            return create_error_response(
                code=1,
                message=errors,
            )

        else:
            account_id = data.get("accountId")
            pvid = data.get("externalUid")
            translator_language_code = data.get("translatorLanguageCode")
            texts = data.get("texts")

    logging.info(f"Processing request for {account_id=}, {pvid=}...")

    # Only translate if not English
    if translator_language_code not in ["en"]:
        translation = translate(
            texts=texts,
            timeout=5,
            from_language=translator_language_code,
        )

    # Create a list to save each text results as one dictionary of 3 sub-dictionaries:
    # text_dict, categories_dict, and entities_dict
    texts_list = list()

    for ind, text in enumerate(texts):
        text_dict = {
            "original": text,
            "sanitized": None,
            "translation": None,
        }

        categories_dict = {
            "multiclass": None,
            "multilabel": list(),
        }

        entities_dict = {ent: list() for ent in ents_target_mapping.values()}

        if translator_language_code not in ["en"]:
            logging.info("Updating text dictionary...")
            text_dict["translation"] = translation[ind]["translations"][0]["text"]

        # Make doc from translation if available
        # Sanitize text as models are trained on sanitized corpus
        if text_dict["translation"] is not None:
            text_sanitized = sanitize(text_dict["translation"])

        else:
            text_sanitized = sanitize(text_dict["original"])

        text_dict["sanitized"] = text_sanitized

        doc = nlp(text_sanitized)

        # textcat
        logging.info("Updating categories dictionary...")

        categories_dict["multiclass"] = max(
            doc.cats,
            key=lambda key: doc.cats[key],
        ).lower()

        categories_dict["multilabel"] = [
            k.lower() for k, v in doc.cats.items() if round(v, 2) > 0.50
        ]

        # ner"
        logging.info("Updating entities dictionary...")

        for ent in doc.ents:
            if ent.label_.lower() in ents_target:
                ent_label = ent.label_.lower()
                ent_label_mapping = ents_target_mapping[ent_label]

                entities_dict[ent_label_mapping].append(ent.text)

        logging.info("Updating texts list...")

        texts_list.append(
            {
                "text": text_dict,
                "categories": categories_dict,
                "entities": entities_dict,
            }
        )

    model_dict = {
        "name": MODEL_NAME,
        "version": MODEL_VERSION,
    }

    environment_dict = {
        "name": ENVIRONMENT_NAME,
        "version": ENVIRONMENT_VERSION,
    }

    response = {
        "texts": texts_list,
        "model": model_dict,
        "environment": environment_dict,
    }

    logging.info(f"{pvid=}: {response=}...")

    # Validate response
    try:
        ResponseModel(**response)

    except ValidationError as error:
        return create_error_response(
            code=0,
            message=error.errors(),
        )

    else:
        response = AMLResponse(
            message=json.dumps(
                response,
                ensure_ascii=False,
            ),
            status_code=200,
        )

        response.mimetype = "application/json"

        return response

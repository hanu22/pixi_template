from typing import Literal

from pydantic import BaseModel, conlist, constr


class RequestModel(BaseModel):
    accountId: constr(min_length=1)
    externalUid: constr(min_length=1)
    translatorLanguageCode: Literal[
        "cs",
        "en",
        "fr",
        "hu",
        "it",
        "nl",
        "nl",
        "pl",
        "ro",
        "sk",
    ]
    texts: conlist(item_type=constr(min_length=1), min_length=1)


class textsResponse(BaseModel):
    text: dict
    categories: dict


class ModelResponse(BaseModel):
    name: constr(min_length=1)
    version: constr(min_length=1)


class environmentResponse(BaseModel):
    name: constr(min_length=1)
    version: constr(min_length=1)


class ResponseModel(BaseModel):
    texts: conlist(item_type=textsResponse, min_length=1)
    model: ModelResponse
    environment: environmentResponse

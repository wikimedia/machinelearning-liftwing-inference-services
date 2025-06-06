import json
import logging
import re
from typing import Any, Union

from kserve.errors import InvalidInput


def is_domain_wikipedia(event: dict) -> bool:
    if "meta" in event and "domain" in event["meta"]:
        return "wikipedia" in event["meta"]["domain"]
    else:
        return False


def check_input_param(**kwargs: dict[str, Any]):
    for key, value in kwargs.items():
        if not value:
            logging.error(f"Missing {key} in input data.")
            raise InvalidInput(f"The parameter {key} is required.")


def get_lang(inputs: dict, event_input_key) -> str:
    try:
        if event_input_key in inputs:
            if inputs[event_input_key]["$schema"].startswith(
                "/mediawiki/revision/create/1"
            ) or inputs[event_input_key]["$schema"].startswith(
                "/mediawiki/revision/create/2"
            ):
                database = inputs[event_input_key]["database"]
                lang = re.match(r"(\w+)wiki", database).group(1)
            elif inputs[event_input_key]["$schema"].startswith(
                "/mediawiki/page/change/1"
            ):
                wiki_id = inputs[event_input_key]["wiki_id"]
                lang = re.match(r"(\w+)wiki", wiki_id).group(1)
            else:
                raise InvalidInput(
                    f"Unsupported event of schema {inputs[event_input_key]['$schema']}, "
                    "the lang value cannot be determined."
                )
        else:
            lang = inputs["lang"]
            if not isinstance(lang, str):
                logging.error("Expected lang to be a string.")
                raise InvalidInput('Expected "lang" to be a string.')
    except KeyError:
        logging.error("Missing lang in input data.")
        raise InvalidInput('Missing "lang" in input data.')
    return lang


def get_page_title(inputs: dict, event_input_key) -> str:
    try:
        if event_input_key in inputs:
            if inputs[event_input_key]["$schema"].startswith(
                "/mediawiki/revision/create/1"
            ) or inputs[event_input_key]["$schema"].startswith(
                "/mediawiki/revision/create/2"
            ):
                page_title = inputs[event_input_key]["page_title"]
            elif inputs[event_input_key]["$schema"].startswith(
                "/mediawiki/page/change/1"
            ):
                page_title = inputs[event_input_key]["page"]["page_title"]
            else:
                raise InvalidInput(
                    f"Unsupported event of schema {inputs[event_input_key]['$schema']}, "
                    "the page_title value cannot be determined."
                )
        else:
            page_title = inputs["page_title"]
            if not isinstance(page_title, str):
                logging.error("Expected page_title to be a string.")
                raise InvalidInput('Expected "page_title" to be a string.')
    except KeyError:
        logging.error("Missing page_title in input data.")
        raise InvalidInput('Missing "page_title" in input data.')
    return page_title


def get_rev_id(inputs: dict, event_input_key) -> int:
    """
    Extracts the revision ID i.e rev_id from an event dictionary.
    This function handles different event schemas related to MediaWiki
    revisions and page changes.
    """
    try:
        if event_input_key in inputs:
            if inputs[event_input_key]["$schema"].startswith(
                "/mediawiki/revision/create/1"
            ) or inputs[event_input_key]["$schema"].startswith(
                "/mediawiki/revision/create/2"
            ):
                rev_id = inputs[event_input_key]["rev_id"]
            elif inputs[event_input_key]["$schema"].startswith(
                "/mediawiki/page/change/1"
            ):
                rev_id = inputs[event_input_key]["revision"]["rev_id"]
            else:
                raise InvalidInput(
                    f"Unsupported event of schema {inputs[event_input_key]['$schema']}, "
                    "the rev_id value cannot be determined."
                )
        else:
            rev_id = inputs["rev_id"]
            if not isinstance(rev_id, int):
                logging.error("Expected rev_id to be an int.")
                raise InvalidInput('Expected "rev_id" to be a int.')
    except KeyError:
        logging.error("Missing rev_id in input data.")
        raise InvalidInput('Missing "rev_id" in input data.')
    return rev_id


def validate_json_input(inputs: Union[dict, bytes]) -> dict:
    """
    Transform inputs to a Dict if inputs are passed as bytes.
    Kserve 0.11 introduced allows any content-type to be passed in the request.
    Since we only use content-type application/json if it isn't provided as a header the body of the
    request is read as bytes. This is added to avoid failures for users that have already been using
    inference services without passing a header.
    """
    if isinstance(inputs, bytes):
        try:
            inputs = inputs.decode("utf-8")
            inputs = json.loads(inputs)
        except (AttributeError, json.decoder.JSONDecodeError):
            raise InvalidInput("Please verify that request input is a json dict")
    return inputs


def check_wiki_suffix(lang: str) -> None:
    if lang.endswith("wiki"):
        raise InvalidInput(
            "Language field should not have a 'wiki' suffix, i.e. use 'en', not 'enwiki'"
        )


def check_supported_wikis(model, lang: str) -> None:
    if hasattr(model, "supported_wikis") and lang not in model.supported_wikis:
        logging.info(f"Unsupported lang: {lang}.")
        raise InvalidInput(f"Unsupported lang: {lang}.")

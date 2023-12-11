import json
import logging
import re
from typing import Any, Dict, Union

from kserve.errors import InvalidInput


def is_domain_wikipedia(event: Dict) -> bool:
    if "meta" in event and "domain" in event["meta"]:
        return "wikipedia" in event["meta"]["domain"]
    else:
        return False


def check_input_param(**kwargs: Dict[str, Any]):
    for key, value in kwargs.items():
        if value is None:
            logging.error(f"Missing {key} in input data.")
            raise InvalidInput(f"The parameter {key} is required.")


def get_lang(inputs: Dict, event_input_key) -> str:
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


def get_page_title(inputs: Dict, event_input_key) -> str:
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


def validate_json_input(inputs: Union[Dict, bytes]) -> Dict:
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

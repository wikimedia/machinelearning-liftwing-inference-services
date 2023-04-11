import re
import logging

from typing import Dict

from kserve.errors import InvalidInput


def get_lang(inputs: Dict, event_input_key) -> Dict:
    try:
        if event_input_key in inputs:
            if inputs[event_input_key]["$schema"].startswith(
                "/mediawiki/revision/create/1"
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


def get_page_title(inputs: Dict, event_input_key) -> Dict:
    try:
        if event_input_key in inputs:
            if inputs[event_input_key]["$schema"].startswith(
                "/mediawiki/revision/create/1"
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

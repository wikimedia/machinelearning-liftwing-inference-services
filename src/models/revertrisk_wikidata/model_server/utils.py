import ast
import json
import logging
import re
from copy import deepcopy
from typing import Any, Optional, Union

import mwapi
import pandas as pd
from deepdiff import DeepDiff

# --- Helper functions from knowledge_integrity library branch wikidata-graph2text ---


# From parsing_utils.py
def _parse_nested_change(old_dict: dict, new_dict: dict) -> Any:
    """
    Parse a nested change between two dictionaries.
    """
    hash_pattern = re.compile(r".*\['hash'\]")
    id_pattern = re.compile(r".*\['id'\]")
    diff = DeepDiff(
        new_dict,
        old_dict,
        ignore_order=True,
        exclude_regex_paths=[hash_pattern, id_pattern],
    )
    return diff.get("values_changed", {})


def _find_key_in_nested_dict(
    haystack: Union[dict, list], target_key: str
) -> Optional[str]:
    """
    Find a key in a nested dictionary or list of dictionaries.
    """
    if isinstance(haystack, dict):
        for key, value in haystack.items():
            if isinstance(value, (dict, list)):
                nested_result = _find_key_in_nested_dict(value, target_key)
                if nested_result is not None:
                    return nested_result
            elif key == target_key:
                return value
    elif isinstance(haystack, list):
        for item in haystack:
            if isinstance(item, (dict, list)):
                nested_result = _find_key_in_nested_dict(item, target_key)
                if nested_result is not None:
                    return nested_result
    return None


def _parse_type_changes(type_changes_dict: dict) -> tuple[dict, dict]:
    """
    Parse type changes from a deepdiff result.
    """
    adds = dict()
    removes = dict()
    list_to_look = ["id", "value", "title"]
    try:
        for key, value in type_changes_dict.items():
            if len(value["old_value"]) == 0 and len(value["new_value"]) != 0:
                property_name = _find_key_in_nested_dict(value["new_value"], "property")
                for k in list_to_look:
                    property_value = _find_key_in_nested_dict(value["new_value"], k)
                    if property_value:
                        break
                if property_name:
                    adds[f"{key}['{property_name}']"] = property_value
                else:
                    adds[key] = property_value
            if len(value["new_value"]) == 0 and len(value["old_value"]) != 0:
                property_name = _find_key_in_nested_dict(value["old_value"], "property")
                for k in list_to_look:
                    property_value = _find_key_in_nested_dict(value["old_value"], k)
                    if property_value:
                        break
                if property_name:
                    removes[f"{key}['{property_name}']"] = property_value
                else:
                    removes[key] = property_value
    except Exception as e:
        logging.warn(
            f"Error while processing type changes: {str(type_changes_dict)}."
            + f"Error: {str(e)}",
        )
    return adds, removes


def _get_element_by_path(obj: Any, path: str) -> str:
    """
    Get an element from a nested object using a path string.
    """
    keys = re.findall(r"\['(.*?)'\]", path)
    current_value = obj
    for key in keys:
        if key in current_value:
            current_value = current_value[key]
        else:
            current_value = None
            break
    return current_value


def parse_wikidata_revision_difference(
    old_revision: Union[str, dict[str, Any]], new_revision: Union[str, dict[str, Any]]
) -> list[str]:
    """
    Parse the difference between two Wikidata revisions.
    """
    try:
        if isinstance(old_revision, str):
            old_revision = json.loads(old_revision)
        if isinstance(new_revision, str):
            new_revision = json.loads(new_revision)

        hash_pattern = re.compile(r"root\['claims'\]\['.*?'\]\[\d*\].*\['hash'\]")
        id_pattern = re.compile(r"root\['claims'\]\['.*?'\]\[\d*\].*\['id'\]")
        diff = DeepDiff(
            old_revision,
            new_revision,
            ignore_order=True,
            exclude_regex_paths=[hash_pattern, id_pattern],
        )

        parsed_diff = {
            "added": diff.get("dictionary_item_added", []),
            "removed": diff.get("dictionary_item_removed", []),
            "type_changes": diff.get("type_changes", {}),
            "iterable_added": diff.get("iterable_item_added", {}),
            "iterable_removed": diff.get("iterable_item_removed", {}),
            "changed": diff.get("values_changed", {}),
        }

        additional_add, additional_removes = _parse_type_changes(
            parsed_diff["type_changes"]
        )

        processed_changes = {}
        keys_to_delete = []
        for key, value in parsed_diff["changed"].items():
            if isinstance(value["new_value"], dict) and isinstance(
                value["old_value"], dict
            ):
                parsed_changes = _parse_nested_change(
                    value["new_value"], value["old_value"]
                )
                keys_to_delete.append(key)
                for key_c, value_c in parsed_changes.items():
                    processed_changes[key + key_c] = value_c
        parsed_diff["changed"].update(processed_changes)
        for k in keys_to_delete:
            del parsed_diff["changed"][k]

        processed_removes = {}
        for key in parsed_diff["removed"]:
            processed_removes[key] = _get_element_by_path(old_revision, key)
        parsed_diff["removed"] = processed_removes
        parsed_diff["removed"].update(parsed_diff["iterable_removed"])
        parsed_diff["removed"].update(additional_removes)

        processed_add = {}
        for key in parsed_diff["added"]:
            processed_add[key] = _get_element_by_path(new_revision, key)
        parsed_diff["added"] = processed_add
        parsed_diff["added"].update(parsed_diff["iterable_added"])
        parsed_diff["added"].update(additional_add)

        return [
            str(parsed_diff["added"]),
            str(parsed_diff["removed"]),
            str(parsed_diff["changed"]),
        ]
    except Exception:
        return [str({}), str({}), str({})]


# From text_utils.py
PATTERN_P = r"P\d+"
PATTERN_Q = r"Q\d+"
DEFAULT_VALUE = "unknown"


def check_id_pattern(string: str) -> bool:
    """
    Check if a string is a Wikidata P-ID or Q-ID.
    """
    if re.fullmatch(PATTERN_P, string) or re.fullmatch(PATTERN_Q, string):
        return True
    else:
        return False


def check_important_wording(string: str) -> bool:
    """
    Check if a string is one of the important wording for text processing.
    """
    if string in [
        "amount",
        "unit",
        "time",
        "timezone",
        "latitude",
        "longitude",
        "altitude",
        "text",
    ]:
        return True
    else:
        return False


def process_key(key: str) -> list[str]:
    """
    Process a key string from a deepdiff path.
    """
    pattern = r"\['(.*?)'\]"
    matches = re.findall(pattern, key)
    if matches:
        return matches
    else:
        return []


def remove_wikilink(link: str) -> str:
    """
    Remove the wikidata entity URL prefix from a link.
    """
    link = str(link)
    return link.replace("http://www.wikidata.org/entity/", "")


def get_value_by_type(json: dict, datatype: str) -> list[str]:
    """
    Get the value from a datavalue object based on its type.
    """
    if datatype == "wikibase-entityid":
        try:
            return [datatype, json["value"]["id"]]
        except Exception:
            return [datatype, DEFAULT_VALUE]
    elif datatype == "string":
        return [datatype, json["value"]]
    elif datatype == "globecoordinate":
        return [
            datatype,
            json["value"]["latitude"],
            json["value"]["longitude"],
            json["value"]["altitude"],
        ]
    elif datatype == "monolingualtext":
        return [datatype, json["value"]["text"]]
    elif datatype == "time":
        return [datatype, json["value"]["time"], json["value"]["timezone"]]
    elif datatype == "quantity":
        return [
            datatype,
            json["value"]["amount"],
            remove_wikilink(json["value"]["unit"]),
        ]
    else:
        return [datatype]


def process_sentence(items: list[str], labels_dict: dict) -> str:
    """
    Process a list of items into a natural language sentence.
    Only Q/P IDs are replaced with their quoted label, others are left as-is.
    """

    def get_label(val):
        if val in labels_dict:
            return labels_dict[val]
        elif check_id_pattern(val):
            return "unknown"
        else:
            return val

    items_transformed = [get_label(i) for i in items]
    return " ".join(items_transformed)


def process_alteration(
    left_q_id: str,
    alterations: str,
    action_type: str = "remove: ",
    labels_dict: dict = {},
) -> list[str]:
    """
    Process added or removed items from a diff.
    """
    initial_sentence = [
        action_type,
        left_q_id if not pd.isna(left_q_id) else DEFAULT_VALUE,
    ]
    v_tmp = ast.literal_eval(alterations)
    sentences = []
    for key in v_tmp.keys():
        sentence_key = deepcopy(initial_sentence)
        sentence_key += process_key(key)
        if "sitelinks" in sentence_key:
            continue
        elif (
            ("aliases" in sentence_key)
            or ("labels" in sentence_key)
            or ("descriptions" in sentence_key)
        ):
            if isinstance(v_tmp[key], list):
                for el in v_tmp[key]:
                    sentence_copy = deepcopy(sentence_key)
                    sentence_copy.append(el["value"])
                    sentences.append(process_sentence(sentence_copy, labels_dict))
            elif isinstance(v_tmp[key], str):
                sentence_copy = deepcopy(sentence_key)
                sentence_copy.append(v_tmp[key])
                sentences.append(process_sentence(sentence_copy, labels_dict))
            else:
                sentence_copy = deepcopy(sentence_key)
                sentence_copy.append(v_tmp[key]["value"])
                sentences.append(process_sentence(sentence_copy, labels_dict))
        elif ("claims" in sentence_key) and (len(sentence_key) <= 4):
            if isinstance(v_tmp[key], list):
                for el in v_tmp[key]:
                    if el["mainsnak"].get("datavalue"):
                        datatype = el["mainsnak"]["datavalue"]["type"]
                        if datatype in [
                            "string",
                            "monolingualtext",
                            "wikibase-entityid",
                        ]:
                            sentence_copy = deepcopy(sentence_key)
                            sentence_copy += get_value_by_type(
                                el["mainsnak"]["datavalue"], datatype
                            )
                            sentences.append(
                                process_sentence(sentence_copy, labels_dict)
                            )
            elif isinstance(v_tmp[key], str):
                sentence_copy = deepcopy(sentence_key)
                sentence_copy.append(v_tmp[key])
                sentences.append(process_sentence(sentence_copy, labels_dict))
            else:
                if v_tmp[key]["mainsnak"].get("datavalue"):
                    datatype = v_tmp[key]["mainsnak"]["datavalue"]["type"]
                    if datatype in ["string", "monolingualtext", "wikibase-entityid"]:
                        sentence_copy = deepcopy(sentence_key)
                        sentence_copy += get_value_by_type(
                            v_tmp[key]["mainsnak"]["datavalue"], datatype
                        )
                        sentences.append(process_sentence(sentence_copy, labels_dict))
    return sentences


def process_change(
    left_q_id: str, changes: str, action_type: str = "change: ", labels_dict: dict = {}
) -> list[tuple[str, str]]:
    """
    Process changed items from a diff.
    """
    initial_sentence = [
        action_type,
        left_q_id if not pd.isna(left_q_id) else DEFAULT_VALUE,
    ]
    v_tmp = ast.literal_eval(changes)
    sentences = []
    for key in v_tmp.keys():
        sentence_key = deepcopy(initial_sentence)
        sentence_key += process_key(key)
        if (
            ("aliases" in sentence_key)
            or ("labels" in sentence_key)
            or ("descriptions" in sentence_key)
        ):
            sentence_copy_old, sentence_copy_new = (
                deepcopy(sentence_key),
                deepcopy(sentence_key),
            )
            sentence_copy_old += [v_tmp[key]["old_value"]]
            sentence_copy_new += [v_tmp[key]["new_value"]]
            sentences.append(
                (
                    process_sentence(sentence_copy_old, labels_dict),
                    process_sentence(sentence_copy_new[1:], labels_dict),
                )
            )
        elif (
            ("claims" in sentence_key)
            and ("qualifiers" not in sentence_key)
            and ("rank" not in sentence_key)
        ):
            items_to_add = [
                i
                for i in sentence_key[4:]
                if check_id_pattern(i) or check_important_wording(i)
            ]
            sentence_copy_old, sentence_copy_new = (
                deepcopy(sentence_key[:4]),
                deepcopy(sentence_key[:4]),
            )
            if sentence_key[-1] == "numeric-id":
                new_value, old_value = (
                    f"Q{v_tmp[key]['new_value']}",
                    f"Q{v_tmp[key]['old_value']}",
                )
            else:
                new_value, old_value = (
                    remove_wikilink(v_tmp[key]["new_value"]),
                    remove_wikilink(v_tmp[key]["old_value"]),
                )
            sentence_copy_old += items_to_add + [old_value]
            sentence_copy_new += items_to_add + [new_value]
            sentence_copy_old, sentence_copy_new = (
                [str(i) for i in sentence_copy_old],
                [str(i) for i in sentence_copy_new],
            )
            sentences.append(
                (
                    process_sentence(sentence_copy_old, labels_dict),
                    process_sentence(sentence_copy_new[1:], labels_dict),
                )
            )
    return sentences


# From bert.py
def prepare_input_for_bert(
    bert_input: Union[list[str], tuple],
) -> Union[list[str], dict]:
    """
    Prepare input for the BERT model.
    """
    bert_input_processed = []
    for element in bert_input:
        if isinstance(element, str):
            bert_input_processed.append(element)
        elif isinstance(element, tuple):
            bert_input_processed.append(
                {"text": element[0], "text_pair": element[1]}  # type: ignore
            )
    return bert_input_processed


def process_transformer_predictions(predictions: list[dict[str, Any]]) -> list[float]:
    """
    Process the predictions from the transformer model.
    """
    scores = []
    for pred in predictions:
        for i in pred:
            if i["label"] == "LABEL_1":
                scores.append(i["score"])
    return scores


async def fetch_labels_from_api(
    session: mwapi.AsyncSession, entity_ids: list[str]
) -> dict[str, str]:
    """
    Fetch labels for a list of Wikidata entity/property IDs using the Wikidata API.
    """
    batch_size = 50
    labels = {}
    for idx in range(0, len(entity_ids), batch_size):
        result = await session.get(
            action="wbgetentities",
            props="labels",
            languages="en",
            ids=entity_ids[idx : idx + batch_size],
        )
        for id, entity in result.get("entities", {}).items():
            label = entity.get("labels", {}).get("en", {}).get("value")
            if label:
                labels[id] = f'"{label}"'
    return labels

import json
from typing import Any

import yaml


def load_data() -> tuple[dict[str, list[dict[str, Any]]], dict[str, tuple[int, int]]]:
    with open("deployed_models.yaml") as f:
        deployed_models = yaml.safe_load(f)
    with open("rev_ids.json") as f:
        rev_ids_list = json.load(f)
    rev_ids_dict = {r["wiki_db"]: (r["rev_id_1"], r["rev_id_2"]) for r in rev_ids_list}
    return deployed_models, rev_ids_dict

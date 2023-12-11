import logging
import pathlib
from itertools import product

import yaml
from utils import load_data

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    deployed_models, rev_ids_dict = load_data()
    tests = {}
    for env in deployed_models:
        tests[env] = {}
        data_centers = ["codfw", "eqiad"] if env == "production" else ["codfw"]
        for model_type in deployed_models[env]:
            for data_center, wiki in product(
                data_centers, model_type["deployed_models"]
            ):
                if wiki in ["eswikibooks", "enwiktionary", "eswikiquote"]:
                    lang = wiki[:2]  # get the language code
                else:
                    lang = wiki
                host = f'https://{wiki}wiki-{model_type["model_name"]}.{model_type["hostname"]}.wikimedia.org'
                path = f'/v1/models/{wiki}wiki-{model_type["model_name"]}:predict'
                # FIXME: for some wikis, we need to use the second rev_id because the first one
                # isn't found on mw api
                if wiki in ["eswikibooks", "enwiktionary", "eswikiquote", "pl"]:
                    rev_id = rev_ids_dict[f"{lang}wiki"][1]
                else:
                    rev_id = rev_ids_dict[f"{lang}wiki"][0]
                json_body = {"rev_id": rev_id}
                tests[env][host] = [
                    {
                        "path": path,
                        "json_body": json_body,
                        "assert_status": 200,
                        "assert_body_contains": "probability",
                        "method": "POST",
                    }
                ]
    for env, values in tests.items():
        with open(f"test_liftwing_{env}.yaml", "w") as f:
            yaml.safe_dump(values, f, sort_keys=False)
        logging.info(
            f"Saved httpbb tests in file {pathlib.Path().resolve()}/test_liftwing_{env}.yaml"
        )

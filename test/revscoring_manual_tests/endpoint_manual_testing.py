import logging
from itertools import product

import requests
from utils import load_data

logging.basicConfig(level=logging.INFO)


def get_response(
    env: str,
    db: str,
    model_name: str,
    rev_id: int,
    model_hostname: str,
    data_center: str = "codfw",
) -> requests.Response:
    url = f"https://inference{'-staging' if env == 'staging' else ''}.svc.{data_center}.wmnet:30443/v1/models/{db}-{model_name}:predict"
    headers = {
        "Content-type": "application/json",
        "Host": f"{db}-{model_name}.{model_hostname}.wikimedia.org",
    }
    data = {"rev_id": rev_id}
    response = requests.post(url, headers=headers, json=data)
    return response


def response_is_ok(resp: requests.Response) -> bool:
    return resp.status_code == 200 and "probability" in resp.text


if __name__ == "__main__":
    deployed_models, rev_ids_dict = load_data()
    errors = []
    for env in deployed_models:
        for model_type in deployed_models[env]:
            data_centers = ["codfw", "eqiad"] if env == "production" else ["codfw"]
            for data_center, wiki in product(
                data_centers, model_type["deployed_models"]
            ):
                if wiki in ["eswikibooks", "enwiktionary", "eswikiquote"]:
                    lang = wiki[:2]  # get the language code
                else:
                    lang = wiki
                logging.info(
                    f"Testing  {env} - {model_type['model_name']} - {wiki} in  {data_center}..."
                )
                endpoint_is_working = False
                for rev_id in rev_ids_dict[f"{lang}wiki"]:
                    response = get_response(
                        env=env,
                        db=f"{wiki}wiki",
                        model_name=model_type["model_name"],
                        rev_id=rev_id,
                        model_hostname=model_type["hostname"],
                        data_center=data_center,
                    )
                    endpoint_is_working = response_is_ok(response)
                    if endpoint_is_working:
                        break
                if not endpoint_is_working:
                    logging.error(
                        f"Endpoint for {env} {model_type['model_name']} {wiki} in {data_center} is not working!"
                    )
                    errors.append([env, model_type["model_name"], wiki, data_center])
                    logging.error(response.text)
    logging.info(f"Errors: {len(errors)} \n {errors}")
    logging.info("All (other) model servers run fine!")

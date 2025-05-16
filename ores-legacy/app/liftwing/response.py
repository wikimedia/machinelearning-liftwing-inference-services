import asyncio
import logging
from typing import List

import aiohttp

from app.utils import create_error_response, manipulate_wp10_call

logger = logging.getLogger(__name__)


@manipulate_wp10_call
async def get_liftwing_response(
    db: str,
    model_name: str,
    rev_id: int,
    features: bool,
    liftwing_url: str,
) -> dict:
    url = f"{liftwing_url}/v1/models/{db}-{model_name}:predict"
    headers = {
        "Content-type": "application/json",
        "Host": f"{db}-{model_name}.{get_lw_namespace(model_name)}.wikimedia.org",
        "User-Agent": "ORES legacy service",
    }
    data = {"rev_id": rev_id, "extended_output": features}
    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(url, headers=headers, json=data) as response:
                logger.debug(
                    f"URL:{url}, HOST:{headers['Host']}, REV_ID:{data['rev_id']}, STATUS: {response.status}"
                )
                if response.status != 200:
                    try:
                        response_json = await response.json()
                        error_message = response_json["error"]
                        print(response_json)
                    except aiohttp.ContentTypeError:
                        error_message = await response.text()
                    except Exception as e:
                        error_message = str(e)
                    logging.error(
                        f"LiftWing call for model {model_name} and rev-id {rev_id} "
                        f"returned {response.status} with message {error_message}"
                    )
                    logging.error(f"Raw Response: {error_message}")
                    return await create_error_response(
                        error_message, response.reason, db, model_name, rev_id
                    )
                return await response.json()
        except aiohttp.ClientConnectorError as e:
            logging.error(
                "The attempt to establish a connection to the LiftWing server raised "
                f"a ClientConnectorError excp with message: {e}"
            )
            return await create_error_response(
                str(e), "ClientConnectorError", db, model_name, rev_id
            )
        except aiohttp.ClientError as e:
            logging.error(
                f"LiftWing call for model {model_name} and rev-id {rev_id} raised "
                f"a ClientError excp with message: {e}"
            )
            return await create_error_response(
                str(e), "ClientError", db, model_name, rev_id
            )


async def make_liftiwing_calls(
    context: str,
    models: List[str],
    rev_ids: List[int],
    features: bool = None,
    liftwing_url: str = "https://inference.discovery.wmnet",
):
    tasks = [
        get_liftwing_response(
            db=context,
            model_name=model,
            rev_id=revid,
            features=features,
            liftwing_url=liftwing_url,
        )
        for revid in rev_ids
        for model in models
    ]
    result = await asyncio.gather(*tasks)
    logger.info(f"Made #{len(result)} calls to LiftWing")
    return result


def get_lw_namespace(model_name: str):
    if model_name in ["damaging", "goodfaith", "reverted"]:
        model_hostname = f"revscoring-editquality-{model_name}"
    elif model_name in ["wp10", "itemquality"]:
        model_hostname = "revscoring-articlequality"
    elif model_name in ["itemtopic"]:
        model_hostname = "revscoring-articletopic"
    else:
        model_hostname = f"revscoring-{model_name}"
    return model_hostname

import asyncio
import logging
import os
import re
import time
from distutils.util import strtobool
from typing import Any, Optional

import aiohttp
import kserve
import mwapi
from kserve.errors import InferenceError, InvalidInput
from utils import ModelLoader, lang_dict

from python.api_utils import get_rest_endpoint_page_summary
from python.preprocess_utils import validate_json_input

logging.basicConfig(level=kserve.constants.KSERVE_LOGLEVEL)


class ArticleDescriptionsModel(kserve.Model):
    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.name = name
        self._http_client_session = {}
        self.date_regex = re.compile("\\d{4}")
        self.aiohttp_client_timeout = os.environ.get("AIOHTTP_CLIENT_TIMEOUT", 5)
        self.user_agent = "WMF ML Team article-description model inference (LiftWing)"
        self.supported_wikipedia_language_codes = list(lang_dict.keys())
        self.model_path = os.environ.get("MODEL_PATH", "/mnt/models/")
        self.wiki_url = os.environ.get("WIKI_URL")
        self.rest_gateway_endpoint = os.environ.get("REST_GATEWAY_ENDPOINT")
        self.low_cpu_mem_usage = strtobool(os.environ.get("LOW_CPU_MEM_USAGE", "True"))
        self.model = ModelLoader()
        self.ready = False
        self.load()

    def load(self) -> None:
        self.model.load_model(self.model_path, self.low_cpu_mem_usage)
        self.ready = True

    async def preprocess(
        self, payload: dict[str, Any], headers: dict[str, str] = None
    ) -> dict[str, Any]:
        payload = validate_json_input(payload)
        lang = payload.get("lang")
        title = payload.get("title")
        num_beams = payload.get("num_beams")
        debug = payload.get("debug", 0)  # default to non debug mode
        self.validate_inputs(lang, title, num_beams, debug)
        execution_times = {}
        features = {}
        starttime = time.time()
        # Retrieve Wikidata info
        descriptions, sitelinks, blp = await self.get_wikidata_info(lang, title)
        execution_times["wikidata-info (s)"] = time.time() - starttime
        features["descriptions"] = descriptions
        # Retrieve first paragraphs
        async with self.get_http_client_session("restgateway") as session:
            tasks = [
                self.get_first_paragraph(lng, sitelinks[lng], session)
                for lng in sitelinks
            ]
            results = await asyncio.gather(*tasks)
        first_paragraphs = dict(zip(sitelinks.keys(), results))

        execution_times["mwapi - first paragraphs (s)"] = time.time() - starttime
        # Retrieve groundtruth description
        groundtruth_desc = await self.get_groundtruth(lang, title)
        execution_times["total network (s)"] = time.time() - starttime
        features["first-paragraphs"] = first_paragraphs
        preprocessed_data = {
            "first_paragraphs": first_paragraphs,
            "descriptions": descriptions,
            "lang": lang,
            "num_beams": num_beams,
            "starttime": starttime,
            "execution_times": execution_times,
            "dates": set(),
            "title": title,
            "blp": blp,
            "groundtruth_desc": groundtruth_desc,
            "features": features,
            "debug": debug,
        }
        return preprocessed_data

    def predict(
        self, preprocessed_data: dict[str, Any], headers: dict[str, str] = None
    ) -> dict[str, Any]:
        first_paragraphs = preprocessed_data["features"]["first-paragraphs"]
        descriptions = preprocessed_data["features"]["descriptions"]
        lang = preprocessed_data["lang"]
        num_beams = preprocessed_data["num_beams"]
        starttime = preprocessed_data["starttime"]
        execution_times = preprocessed_data["execution_times"]
        dates = preprocessed_data["dates"]
        title = preprocessed_data["title"]
        blp = preprocessed_data["blp"]
        groundtruth_desc = preprocessed_data["groundtruth_desc"]
        features = preprocessed_data["features"]
        debug = preprocessed_data["debug"]
        prediction = self.model.predict(
            first_paragraphs,
            descriptions,
            lang,
            num_beams=num_beams,
            num_return_sequences=num_beams,
        )
        execution_times["model (s)"] = (
            time.time() - starttime - execution_times["total network (s)"]
        )
        for desc in descriptions:
            dates.update(self.date_regex.findall(desc))
        for parag in first_paragraphs:
            dates.update(self.date_regex.findall(parag))
        filtered_predictions = []
        for pred in prediction:
            dates_in_p = self.date_regex.findall(pred)
            keep = True
            for date in dates_in_p:
                if date not in dates:
                    keep = False
                    break
            if keep:
                filtered_predictions.append(pred)
        execution_times["total (s)"] = time.time() - starttime
        prediction_data = {
            "lang": lang,
            "title": title,
            "blp": blp,
            "num_beams": num_beams,
            "groundtruth": groundtruth_desc,
            "latency": execution_times,
            "features": features,
            "prediction": prediction,
        }
        # only include minimal fields when debug is not 1
        if not debug:
            prediction_data = {
                key: prediction_data[key]
                for key in ["prediction", "blp", "lang", "title", "num_beams"]
            }
        return prediction_data

    async def get_groundtruth(self, lang: str, title: str) -> str:
        """Get existing article description (groundtruth)."""
        mw_host, host_header = self.get_mw_host_and_header(lang)
        session = mwapi.AsyncSession(
            host=self.wiki_url or mw_host,
            user_agent=self.user_agent,
            session=self.get_http_client_session("mwapi"),
        )
        session.headers["Host"] = host_header
        # English has a prop that takes into account shortdescs (local override) that other languages don't
        if lang == "en":
            try:
                result = await session.get(
                    action="query",
                    prop="pageprops",
                    titles=title,
                    redirects="",
                    format="json",
                    formatversion=2,
                )
                return result["query"]["pages"][0]["pageprops"]["wikibase-shortdesc"]
            except Exception as e:
                logging.error(
                    f"Failed to retrieve groundtruth description for English article: {title}. Reason: {e}"
                )
                return None
        # Non-English languages: get description from Wikidata
        else:
            try:
                result = await session.get(
                    action="query",
                    prop="pageterms",
                    titles=title,
                    redirects="",
                    wbptterms="description",
                    wbptlanguage=lang,
                    format="json",
                    formatversion=2,
                )
                return result["query"]["pages"][0]["terms"]["description"][0]
            except Exception as e:
                logging.error(
                    f"Failed to retrieve groundtruth description for non-English article: {title}. Reason: {e}"
                )
                return None

    async def get_first_paragraph(
        self, lang: str, title: str, session: aiohttp.ClientSession
    ) -> str:
        """Get plain-text extract of article"""
        mw_host, host_header = self.get_mw_host_and_header(lang)
        rest_url = get_rest_endpoint_page_summary(
            self.rest_gateway_endpoint, mw_host, host_header, title
        )
        try:
            async with session.get(
                rest_url,
                headers={
                    "User-Agent": self.user_agent,
                },
            ) as resp:
                paragraph = await resp.json()
                return paragraph["extract"]
        except Exception as e:
            logging.error(f"Failed to retrieve first paragraph: {e}")
            return ""

    async def get_wikidata_info(self, lang: str, title: str) -> dict[str, Any]:
        """Get article descriptions from Wikidata"""
        mw_host, host_header = self.get_mw_host_and_header()
        session = mwapi.AsyncSession(
            host=self.wiki_url or mw_host,
            user_agent=self.user_agent,
            session=self.get_http_client_session("mwapi"),
        )
        session.headers["Host"] = host_header
        try:
            result = await session.get(
                action="wbgetentities",
                sites=f"{lang}wiki",
                titles=title,
                redirects="yes",
                props="descriptions|claims|sitelinks",
                languages="|".join(self.supported_wikipedia_language_codes),
                sitefilter="|".join(
                    [f"{lng}wiki" for lng in self.supported_wikipedia_language_codes]
                ),
                format="json",
                formatversion=2,
            )
            descriptions = {}
            sitelinks = {}
            blp = False
            # should be exactly 1 QID for the page if it has a Wikidata item
            qid = list(result["entities"].keys())[0]
            # get all the available descriptions in relevant languages
            for lng in result["entities"][qid]["descriptions"]:
                descriptions[lng] = result["entities"][qid]["descriptions"][lng][
                    "value"
                ]
            # get the sitelinks from supported languages
            for wiki in result["entities"][qid]["sitelinks"]:
                lang = wiki[:-4]  # remove 'wiki' part
                sitelinks[lang] = result["entities"][qid]["sitelinks"][wiki]["title"]
            human = False
            claims = result["entities"][qid]["claims"]
            for io_claim in claims.get("P31", []):
                if io_claim["mainsnak"]["datavalue"]["value"]["id"] == "Q5":
                    human = True
                    break
            died = "P570" in claims  # date-of-death property
            if human and not died:
                blp = True
        except Exception as e:
            error_message = f"Failed to lookup Wikidata info for \
                page {title}. Reason: {e}"
            logging.error(error_message)
            raise InferenceError(error_message)
        return descriptions, sitelinks, blp

    @staticmethod
    def get_mw_host_and_header(lang: Optional[str] = None) -> (str, str):
        """Get the MediaWiki host for the given language"""
        if not lang:
            return "https://www.wikidata.org", "www.wikidata.org"
        return f"https://{lang}.wikipedia.org", f"{lang}.wikipedia.org"

    def get_http_client_session(self, endpoint):
        """
        Returns a aiohttp session for the specific endpoint passed as input.
        We need to do it since sharing a single session leads to unexpected
        side effects (like sharing headers, most notably the Host one).
        """
        timeout = aiohttp.ClientTimeout(total=self.aiohttp_client_timeout)
        if (
            self._http_client_session.get(endpoint, None) is None
            or self._http_client_session[endpoint].closed
        ):
            logging.info(f"Opening a new Asyncio session for {endpoint}.")
            self._http_client_session[endpoint] = aiohttp.ClientSession(
                timeout=timeout, raise_for_status=True
            )
        return self._http_client_session[endpoint]

    def validate_inputs(
        self, lang: str, title: str, num_beams: int, debug: int
    ) -> None:
        """Validate the user inputs"""
        if not lang:
            error_message = "The parameter lang is required."
            logging.error(error_message)
            raise InvalidInput(error_message)
        elif lang not in self.supported_wikipedia_language_codes:
            error_message = f"Unsupported lang: {lang}. \
                The supported ones are: {self.supported_wikipedia_language_codes}."
            logging.error(error_message)
            raise InvalidInput(error_message)
        if not title:
            error_message = "The parameter title is required."
            logging.error(error_message)
            raise InvalidInput(error_message)
        if not num_beams:
            error_message = "The parameter num_beams is required."
            logging.error(error_message)
            raise InvalidInput(error_message)
        elif not isinstance(num_beams, int) or num_beams < 1:
            error_message = "The parameter num_beams should be a positive number."
            logging.error(error_message)
            raise InvalidInput(error_message)
        if debug not in [0, 1]:
            error_message = "The parameter debug should be either 0 or 1."
            logging.error(error_message)
            raise InvalidInput(error_message)


if __name__ == "__main__":
    model_name = os.environ.get("MODEL_NAME")
    model = ArticleDescriptionsModel(model_name)
    kserve.ModelServer(workers=1).start([model])

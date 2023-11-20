import logging
import os
import re
import time
from typing import Any, Dict

import aiohttp
import kserve
import mwapi

from kserve.errors import InvalidInput, InferenceError
from utils import LANG_DICT, ModelLoader

logging.basicConfig(level=kserve.constants.KSERVE_LOGLEVEL)


class ArticleDescriptionsModel(kserve.Model):
    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.name = name
        self._http_client_session = {}
        self.DATE_REGEX = re.compile("\\d{4}")
        self.AIOHTTP_CLIENT_TIMEOUT = os.environ.get("AIOHTTP_CLIENT_TIMEOUT", 5)
        self.USER_AGENT = "WMF ML Team article-description model inference (LiftWing)"
        self.SUPPORTED_WIKIPEDIA_LANGUAGE_CODES = list(LANG_DICT.keys())
        self.MODEL = ModelLoader()
        self.ready = False
        self.load()

    def load(self) -> None:
        model_path = "/mnt/models/"
        self.MODEL.load_model(model_path)
        self.ready = True

    async def preprocess(
        self, payload: Dict[str, Any], headers: Dict[str, str] = None
    ) -> Dict[str, Any]:
        lang = payload.get("lang")
        title = payload.get("title")
        num_beams = payload.get("num_beams")
        self.validate_inputs(lang, title, num_beams)
        execution_times = {}
        features = {}
        starttime = time.time()
        # Retrieve Wikidata info
        descriptions, sitelinks, blp = await self.get_wikidata_info(lang, title)
        wd_time = time.time()
        execution_times["wikidata-info (s)"] = wd_time - starttime
        features["descriptions"] = descriptions
        # Retrieve first paragraphs
        first_paragraphs = {}
        for lng in sitelinks:
            first_paragraph = await self.get_first_paragraph(lng, sitelinks[lng])
            first_paragraphs[lng] = first_paragraph
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
        }
        return preprocessed_data

    def predict(
        self, preprocessed_data: Dict[str, Any], headers: Dict[str, str] = None
    ) -> Dict[str, Any]:
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
        prediction = self.MODEL.predict(
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
            dates.update(self.DATE_REGEX.findall(desc))
        for parag in first_paragraphs:
            dates.update(self.DATE_REGEX.findall(parag))
        filtered_predictions = []
        for pred in prediction:
            dates_in_p = self.DATE_REGEX.findall(pred)
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
        return prediction_data

    async def get_groundtruth(self, lang: str, title: str) -> str:
        """Get existing article description (groundtruth)."""
        session = mwapi.AsyncSession(
            host=f"https://{lang}.wikipedia.org",
            user_agent=self.USER_AGENT,
            session=self.get_http_client_session("mwapi"),
        )
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

    async def get_first_paragraph(self, lang: str, title: str) -> str:
        """Get plain-text extract of article"""
        try:
            base_url = f"https://{lang}.wikipedia.org"
            async with self.get_http_client_session(base_url) as session:
                async with session.get(
                    f"{base_url}/api/rest_v1/page/summary/{title}",
                    headers={"User-Agent": self.USER_AGENT},
                ) as resp:
                    paragraph = await resp.json()
                    return paragraph["extract"]
        except Exception as e:
            logging.error(f"Failed to retrieve first paragraph: {e}")
            return ""

    async def get_wikidata_info(self, lang: str, title: str) -> Dict[str, Any]:
        """Get article descriptions from Wikidata"""
        session = mwapi.AsyncSession(
            host="https://wikidata.org",
            user_agent=self.USER_AGENT,
            session=self.get_http_client_session("mwapi"),
        )
        try:
            result = await session.get(
                action="wbgetentities",
                sites=f"{lang}wiki",
                titles=title,
                redirects="yes",
                props="descriptions|claims|sitelinks",
                languages="|".join(self.SUPPORTED_WIKIPEDIA_LANGUAGE_CODES),
                sitefilter="|".join(
                    [f"{lng}wiki" for lng in self.SUPPORTED_WIKIPEDIA_LANGUAGE_CODES]
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
            logging.error(
                f"Failed to lookup Wikidata info for page {title}. Reason: {e}"
            )
            raise InferenceError(
                "An error occurred while fetching info for title "
                "from the Wikidata API, please contact the ML-Team "
                "if the issue persists."
            )
        return descriptions, sitelinks, blp

    def get_http_client_session(self, endpoint):
        """
        Returns a aiohttp session for the specific endpoint passed as input.
        We need to do it since sharing a single session leads to unexpected
        side effects (like sharing headers, most notably the Host one).
        """
        timeout = aiohttp.ClientTimeout(total=self.AIOHTTP_CLIENT_TIMEOUT)
        if (
            self._http_client_session.get(endpoint, None) is None
            or self._http_client_session[endpoint].closed
        ):
            logging.info(f"Opening a new Asyncio session for {endpoint}.")
            self._http_client_session[endpoint] = aiohttp.ClientSession(
                timeout=timeout, raise_for_status=True
            )
        return self._http_client_session[endpoint]

    def validate_inputs(self, lang: str, title: str, num_beams: int) -> None:
        """Validate the user inputs"""
        if not lang:
            logging.error("Missing lang in input data.")
            raise InvalidInput("The parameter lang is required.")
        elif lang not in self.SUPPORTED_WIKIPEDIA_LANGUAGE_CODES:
            logging.error(
                f"Unsupported lang: {lang}. \
                The supported ones are: {self.SUPPORTED_WIKIPEDIA_LANGUAGE_CODES}."
            )
            raise InvalidInput(
                f"Unsupported lang: {lang}. \
                The supported ones are: {self.SUPPORTED_WIKIPEDIA_LANGUAGE_CODES}."
            )
        if not title:
            logging.error("Missing title in input data.")
            raise InvalidInput("The parameter title is required.")
        if not num_beams:
            logging.error("Missing num_beams in input data.")
            raise InvalidInput("The parameter num_beams is required.")
        elif not isinstance(num_beams, int) or num_beams < 1:
            logging.error("num_beams in input data should be a positive number.")
            raise InvalidInput("The parameter num_beams should be a positive number.")


if __name__ == "__main__":
    model_name = os.environ.get("MODEL_NAME")
    model = ArticleDescriptionsModel(model_name)
    kserve.ModelServer(workers=1).start([model])
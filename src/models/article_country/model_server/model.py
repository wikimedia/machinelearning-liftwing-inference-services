import logging
import os
from distutils.util import strtobool
from typing import Any, Dict

import kserve
from kserve.errors import InvalidInput

from python.preprocess_utils import (
    check_input_param,
    check_wiki_suffix,
    validate_json_input,
)
from utils import (
    get_cultural_countries,
    get_claims,
    get_geographic_country,
    load_categories,
    load_country_aggregations,
    load_country_properties,
    load_countries_data,
    load_geometries,
    title_to_categories,
    title_to_qid,
    validate_qid,
)


logging.basicConfig(level=kserve.constants.KSERVE_LOGLEVEL)


class ArticleCountryModel(kserve.Model):
    def __init__(self, name: str, data_path: str, force_http: bool) -> None:
        super().__init__(name)
        self.name = name
        self.protocol = "http" if force_http else "https"
        self.data_path = data_path
        self.ready = False
        self.load()

    def load(self) -> None:
        # Load pre-computed data to be used for geospatial analysis
        self.country_properties = load_country_properties("country_properties.tsv")
        self.qid_to_region = load_countries_data("countries.tsv")
        self.qid_to_region = load_country_aggregations(
            self.qid_to_region, "country_aggregation.tsv"
        )
        self.qid_to_geometry = load_geometries(
            self.data_path, "ne_10m_admin_0_map_units.geojson", self.qid_to_region
        )
        self.category_to_country = load_categories(
            self.data_path, "category-countries.tsv.gz", self.qid_to_region
        )
        self.ready = True

    async def preprocess(
        self, inputs: Dict[str, Any], headers: Dict[str, str] = None
    ) -> Dict[str, Any]:
        inputs = validate_json_input(inputs)
        qid = inputs.get("qid")
        lang = inputs.get("lang")
        title = inputs.get("title")
        check_input_param(lang=lang, title=title)
        check_wiki_suffix(lang)

        if qid:
            qid = qid.upper()
            if not validate_qid(qid):
                error_message = (
                    f"Poorly formatted 'qid' field. '{qid}' does not match '^Q[0-9]+$'"
                )
                logging.error(error_message)
                raise InvalidInput(error_message)
        else:
            qid = title_to_qid(lang, title, self.protocol)

        claims = get_claims(self.protocol, qid)
        country_categories = title_to_categories(
            title,
            lang,
            self.protocol,
            self.category_to_country,
        )
        peprocessed_data = {
            "qid": qid,
            "claims": claims,
            "country_categories": country_categories,
        }
        return peprocessed_data

    def predict(
        self, request: Dict[str, Any], headers: Dict[str, str] = None
    ) -> Dict[str, Any]:
        claims = request.get("claims")
        country_categories = request.get("country_categories")
        prediction = {"qid": request.get("qid"), "countries": [], "wikidata": []}
        countries = set()
        details = []
        for property, p_country in get_cultural_countries(
            claims, self.country_properties, self.qid_to_region
        ):
            details.append(
                {property: self.country_properties[property], "country": p_country}
            )
            countries.add(p_country)
        country = get_geographic_country(
            claims, self.qid_to_geometry, self.qid_to_region
        )
        if country:
            details.append({"P625": "coordinate location", "country": country})
            countries.add(country)
        prediction["wikidata"] = details
        prediction["categories"] = []
        for country in country_categories:
            countries.add(country)
            prediction["categories"].append(
                {
                    "country": country,
                    "categories": "|".join(country_categories[country]),
                }
            )
        prediction["countries"] = sorted(list(countries))
        return prediction


if __name__ == "__main__":
    model_name = os.environ.get("MODEL_NAME")
    data_path = os.environ.get("DATA_PATH", "/mnt/models/")
    force_http = strtobool(os.environ.get("FORCE_HTTP", "False"))
    model = ArticleCountryModel(
        name=model_name, data_path=data_path, force_http=force_http
    )
    kserve.ModelServer(workers=1).start([model])

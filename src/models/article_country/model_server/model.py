import logging
import os
from distutils.util import strtobool
from typing import Any, Dict

import kserve
from aiohttp import ClientSession, ClientTimeout
from kserve.errors import InvalidInput
from python import events
from python.preprocess_utils import (
    check_input_param,
    check_wiki_suffix,
    get_lang,
    get_page_title,
    is_domain_wikipedia,
    validate_json_input,
)
from utils import (
    calculate_sums,
    get_cultural_countries,
    get_claims,
    get_geographic_country,
    init_groundtruth_db,
    load_categories,
    load_country_aggregations,
    load_country_IDFs,
    load_country_properties,
    load_countries_data,
    load_geometries,
    normalize_sums,
    sort_results_by_score,
    title_to_categories,
    title_to_links,
    title_to_qid,
    update_scores,
)


logging.basicConfig(level=kserve.constants.KSERVE_LOGLEVEL)


class ArticleCountryModel(kserve.Model):
    def __init__(
        self, name: str, data_path: str, force_http: bool, aiohttp_client_timeout: int
    ) -> None:
        super().__init__(name)
        self.name = name
        self.protocol = "http" if force_http else "https"
        self.data_path = data_path
        self.event_key = "event"
        self.eventgate_url = os.environ.get("EVENTGATE_URL")
        self.eventgate_prediction_classification_change_stream = os.environ.get(
            "EVENTGATE_PREDICTION_CLASSIFICATION_CHANGE_STREAM"
        )
        self.eventgate_weighted_tags_change_stream = os.environ.get(
            "EVENTGATE_WEIGHTED_TAGS_CHANGE_STREAM"
        )
        # Deployed via the wmf-certificates package
        self.tls_cert_bundle_path = "/etc/ssl/certs/wmf-ca-certificates.crt"
        self.custom_user_agent = (
            "WMF ML Team article-country model inference (LiftWing)"
        )
        self.ready = False
        self.http_client_session = {}
        self.aiohttp_client_timeout = aiohttp_client_timeout
        self.groundtruth_db = None
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
        self.country_IDFs = load_country_IDFs("country_IDFs.tsv")
        # initialize the wikilink groundtruth sqlite database
        self.groundtruth_db = init_groundtruth_db(
            self.data_path,
            "region-groundtruth-2025-01-01-qids.sqlite",
        )
        self.ready = True

    async def preprocess(
        self, inputs: Dict[str, Any], headers: Dict[str, str] = None
    ) -> Dict[str, Any]:
        inputs = validate_json_input(inputs)
        if self.event_key in inputs:
            # process inputs from source event stream
            source_event = inputs.get(self.event_key)
            if not is_domain_wikipedia(source_event):
                error_message = "This model is not recommended for use in projects outside of Wikipedia (e.g. Wiktionary, Wikinews, etc)"
                logging.error(error_message)
                raise InvalidInput(error_message)
            lang = get_lang(inputs, self.event_key)
            title = get_page_title(inputs, self.event_key)
        else:
            # avoid using get_page_title() as it expects "page_title" instead of just "title" in the inputs
            lang = inputs.get("lang")
            title = inputs.get("title")
        check_input_param(lang=lang, title=title)
        check_wiki_suffix(lang)
        qid = await title_to_qid(
            lang,
            title,
            self.protocol,
            self.get_http_client_session("mwapi"),
        )
        claims = await get_claims(
            self.protocol, self.get_http_client_session("mwapi"), qid
        )
        country_categories = await title_to_categories(
            title,
            lang,
            self.protocol,
            self.category_to_country,
            self.get_http_client_session("mwapi"),
        )
        preprocessed_data = {
            "lang": lang,
            "title": title,
            "qid": qid,
            "claims": claims,
            "country_categories": country_categories,
        }
        if self.event_key in inputs:
            preprocessed_data[self.event_key] = inputs.get(self.event_key)
        # preprocess wikilinks based on the groundtruth db
        if self.groundtruth_db:
            link_countries = await title_to_links(
                title,
                lang,
                self.protocol,
                self.get_http_client_session("mwapi"),
                self.groundtruth_db,
                limit=500,
            )
            tfidf_sum = 0
            # temporary dictionary for computed raw tfidf per country
            computed_tfidf = {}
            links_analyzed = sum(link_countries.values())
            # first compute un-normalized tfidf scores for each country
            for country in sorted(link_countries, key=link_countries.get, reverse=True):
                if not country:
                    continue  # skip empty country keys (see T385970#10633019)
                # look up the IDF value (using the empty-string fallback if a country is not found)
                idf_value = self.country_IDFs.get(country, self.country_IDFs.get(""))
                prop_tfidf = (
                    (link_countries[country] / links_analyzed) * idf_value
                    if links_analyzed
                    else 0
                )
                computed_tfidf[country] = prop_tfidf
                tfidf_sum += prop_tfidf
            preprocessed_data["wikilinks"] = {
                "link_countries": link_countries,
                "computed_tfidf": computed_tfidf,
                "tfidf_sum": tfidf_sum,
            }
        return preprocessed_data

    async def predict(
        self, request: Dict[str, Any], headers: Dict[str, str] = None
    ) -> Dict[str, Any]:
        # -------------------------------------------------------------
        # Get values from preprocess and define a prediction structure
        # -------------------------------------------------------------
        claims = request.get("claims")
        country_categories = request.get("country_categories")
        wikilinks_data = request.get("wikilinks")
        link_countries = wikilinks_data.get("link_countries")
        computed_tfidf = wikilinks_data.get("computed_tfidf")
        tfidf_sum = wikilinks_data.get("tfidf_sum")
        prediction = {
            "model_name": self.name,
            "model_version": "1",  # proper versioning will happen when model is available
            "prediction": {
                "article": f'https://{request.get("lang")}.wikipedia.org/wiki/{request.get("title")}',
                "wikidata_item": request.get("qid"),
                "results": [],
            },
        }

        # -------------------------------------------------------------
        # Aggregate results from Wikidata properties, categories, and
        # wikilinks into a country_results dictionary
        # -------------------------------------------------------------
        country_results = {}
        # aggregate Wikidata properties for each country
        for wikidata_property, country in get_cultural_countries(
            claims, self.country_properties, self.qid_to_region
        ):
            if country not in country_results:
                country_results[country] = {
                    "score": 0,
                    "source": {"wikidata_properties": [], "categories": []},
                }
            country_results[country]["source"]["wikidata_properties"].append(
                {wikidata_property: self.country_properties[wikidata_property].get(0)}
            )

        # check geographic country for additional contributions to score
        geographic_country = get_geographic_country(
            claims, self.qid_to_geometry, self.qid_to_region
        )
        if geographic_country:
            if geographic_country not in country_results:
                country_results[geographic_country] = {
                    "score": 0,
                    "source": {"wikidata_properties": [], "categories": []},
                }
            country_results[geographic_country]["source"]["wikidata_properties"].append(
                {"P625": "coordinate location"}
            )

        # add categories to country results
        for country, categories in country_categories.items():
            if country not in country_results:
                country_results[country] = {
                    "score": 0,
                    "source": {"wikidata_properties": [], "categories": []},
                }
            country_results[country]["source"]["categories"].extend(categories)

        # iterate over computed tfidf values to determine normalized wikilink support
        for country, prop_tfidf in computed_tfidf.items():
            normalized_tfidf = (prop_tfidf / tfidf_sum) if tfidf_sum else 0
            # only consider merging if above the thresholds
            if normalized_tfidf >= 0.25 and link_countries[country] >= 3:
                if country in country_results:
                    country_results[country]["source"].setdefault("links", [])
                    country_results[country]["source"]["links"].append(
                        {
                            "country": country,
                            "count": link_countries[country],
                            "prop-tfidf": normalized_tfidf,
                        }
                    )
                else:
                    # create a new country result entry for a country that hasn't been seen
                    country_results[country] = {
                        "source": {
                            "wikidata_properties": [],
                            "categories": [],
                            "links": [
                                {
                                    "country": country,
                                    "count": link_countries[country],
                                    "prop-tfidf": normalized_tfidf,
                                }
                            ],
                        }
                    }

        # -------------------------------------------------------------
        # Convert aggregated results into the final prediction list and
        # compute final scores for each country
        # -------------------------------------------------------------
        for country, result in country_results.items():
            prediction["prediction"]["results"].append(
                {"country": country, "score": 1, "source": result["source"]}
            )
        if prediction["prediction"]["results"]:
            # normalize score based on categories and properties if results exist
            sums = calculate_sums(prediction)
            normalized_scores = normalize_sums(sums)
            update_scores(prediction, normalized_scores)
            prediction = sort_results_by_score(prediction)

        # -------------------------------------------------------------
        # Send prediction events if applicable based on event stream
        # input, then return a prediction reponse
        # -------------------------------------------------------------
        if self.event_key in request and prediction["prediction"]["results"]:
            prediction_results = {
                "predictions": [
                    result["country"] for result in prediction["prediction"]["results"]
                ],  # list of countries in the prediction
                "probabilities": {
                    result["country"]: result["score"]
                    for result in prediction.get("prediction", {}).get("results", [])
                },  # dict of countries and their score from the prediction
            }
            await self.send_prediction_classification_change_event(
                request.get(self.event_key),
                prediction_results,
                prediction.get("model_version"),
            )
            tags_to_set = {
                "classification.prediction.articlecountry": [
                    {"tag": result["country"], "score": result["score"]}
                    for result in prediction["prediction"]["results"]
                ]
            }
            await self.send_weighted_tags_change_event(
                request.get(self.event_key), tags_to_set
            )
        return prediction

    async def send_prediction_classification_change_event(
        self,
        page_change_event: Dict[str, Any],
        prediction_results: Dict[str, Any],
        model_version: str,
    ) -> None:
        """
        Send an article_country prediction_classification_change event to EventGate,
        generated from the page_change event and prediction_results passed as input.
        """
        article_country_prediction_classification_change_event = (
            events.generate_prediction_classification_event(
                page_change_event,
                self.eventgate_prediction_classification_change_stream,
                "article-country",
                model_version,
                prediction_results,
            )
        )
        await events.send_event(
            article_country_prediction_classification_change_event,
            self.eventgate_url,
            self.tls_cert_bundle_path,
            self.custom_user_agent,
            self.get_http_client_session("eventgate"),
        )

    async def send_weighted_tags_change_event(
        self,
        page_change_event: Dict[str, Any],
        tags_to_set: Dict[str, Any],
    ) -> None:
        """
        Send a cirrussearch page_weighted_tags_change event to EventGate, generated
        from the page_change event and prediction results formatted to match the
        shape of tags_to_set.
        """
        article_country_weighted_tags_change_event = (
            events.generate_page_weighted_tags_event(
                page_change_event,
                self.eventgate_weighted_tags_change_stream,
                tags_to_set,
            )
        )
        await events.send_event(
            article_country_weighted_tags_change_event,
            self.eventgate_url,
            self.tls_cert_bundle_path,
            self.custom_user_agent,
            self.get_http_client_session("eventgate"),
        )

    def get_http_client_session(self, endpoint: str) -> ClientSession:
        """
        Returns a aiohttp session for the specific endpoint passed as input.
        We need to do it since sharing a single session leads to unexpected
        side effects (like sharing headers, most notably the Host one).
        """
        timeout = ClientTimeout(total=self.aiohttp_client_timeout)
        if (
            self.http_client_session.get(endpoint, None) is None
            or self.http_client_session[endpoint].closed
        ):
            logging.info(f"Opening a new Asyncio session for {endpoint}.")
            self.http_client_session[endpoint] = ClientSession(
                timeout=timeout, raise_for_status=True
            )
        return self.http_client_session[endpoint]


if __name__ == "__main__":
    model_name = os.environ.get("MODEL_NAME")
    data_path = os.environ.get("DATA_PATH", "/mnt/models/")
    force_http = strtobool(os.environ.get("FORCE_HTTP", "False"))
    aiohttp_client_timeout = os.environ.get("AIOHTTP_CLIENT_TIMEOUT", 5)
    model = ArticleCountryModel(
        name=model_name,
        data_path=data_path,
        force_http=force_http,
        aiohttp_client_timeout=aiohttp_client_timeout,
    )
    kserve.ModelServer(workers=1).start([model])

import json
import logging
import os
import re

import mwapi
import pandas as pd
from kserve.errors import InferenceError
from shapely.geometry import Point, shape
from typing import List, Dict, Optional

logging.basicConfig(level=logging.DEBUG)

current_file_directory = os.path.dirname(__file__)

custom_ua = "WMF ML Team article-country model inference (LiftWing)"

"""
TODO: when the event stream exists, replace SQLite database usage with search index
to make a circular dependency prediction using wikilinks. Reference:
1. https://github.com/wikimedia/research-api-endpoint-template/blob/33b263971f1186befd0d544b59b8121236b5fa62/model/wsgi.py#L43-L49
2. https://phabricator.wikimedia.org/P69329#277631
"""


def get_mediawiki_base_url(protocol: str, lang: Optional[str] = None) -> str:
    """
    Modifies the mediawiki base URL to use the set protocol (http or https)
    """
    if lang:
        base_url = f"{protocol}://{lang}.wikipedia.org"
    else:
        base_url = f"{protocol}://www.wikidata.org"
    return base_url


def validate_qid(qid: str) -> bool:
    """
    Validate that a QID has the format of a Wikidata item ID, which is 'Q' followed by
    one or more digits e.g. "Q12345"
    """
    return bool(re.match("^Q[0-9]+$", qid))


def title_to_qid(lang: str, title: str, protocol: str) -> Optional[str]:
    """
    Get Wikidata item ID(s) for Wikipedia article(s)

    TODO: use asynchronous API calls after testing force_http on staging
    Remember to reuse sessions similar to other model servers.
    """
    session = mwapi.Session(
        get_mediawiki_base_url(protocol, lang), user_agent=custom_ua
    )

    try:
        result = session.get(
            action="query",
            prop="pageprops",
            ppprop="wikibase_item",
            redirects=True,
            titles=title,
            format="json",
            formatversion=2,
        )
        pages = result.get("query", {}).get("pages", [])
        if pages:
            return pages[0].get("pageprops", {}).get("wikibase_item")
    except Exception as e:
        error_message = (
            f"Failed to get wikidata item ID for article title: {title}. Reason: {e}"
        )
        logging.error(error_message)
        raise InferenceError(error_message)

    return None


def get_cultural_countries(
    claims: Dict, country_properties: Dict, qid_to_region: Dict
) -> List[tuple]:
    """
    Extracts cultural regions/countries from the provided Wikidata claims.
    """
    regions = []
    for prop in country_properties:
        if prop in claims:
            for statement in claims[prop]:
                try:
                    value = statement["mainsnak"]["datavalue"]["value"]["id"]
                    if value in qid_to_region:
                        regions.append((prop, qid_to_region[value]))
                except Exception as e:
                    error_message = f"Failed to get cultural regions. Reason: {e}"
                    logging.error(error_message)
                    raise InferenceError(error_message)
    return regions


def get_geographic_country(
    claims: Dict, qid_to_geometry: Dict, qid_to_region: Dict
) -> Optional[str]:
    """
    Checks if the Wikidata claims contain geographic coordinates
    (Q625 is the Wikidata property for coordinate location)
    and returns the country after checking that the coordinates
    are on Earth. Otherwise, it returns None
    """
    if "P625" in claims:
        try:
            coordinates = claims["P625"][0]["mainsnak"]["datavalue"]["value"]
            if (
                coordinates["globe"] == "http://www.wikidata.org/entity/Q2"
            ):  # don't geolocate moon craters etc.
                lat = coordinates["latitude"]
                lon = coordinates["longitude"]
                country = point_in_country(lon, lat, qid_to_geometry, qid_to_region)
                return country
        except Exception as e:
            error_message = f"Failed to get geographic region. Reason: {e}"
            logging.error(error_message)
            raise InferenceError(error_message)
    return None


def point_in_country(
    lon: float, lat: float, qid_to_geometry: Dict, qid_to_region: Dict
) -> str:
    """
    Determine which region contains a lat-lon coordinate.

    Depends on shapely library and region_shapes object, which contains a dictionary
    mapping QIDs to shapely geometry objects.
    """
    pt = Point(lon, lat)
    for qid in qid_to_geometry:
        if qid_to_geometry[qid].contains(pt):
            return qid_to_region[qid]
    return None


def get_claims(protocol: str, qid: Optional[str] = None) -> Dict:
    """
    Get claims from wikibase entity item of provided QID.

    TODO: use asynchronous API calls after testing force_http on staging
    Remember to reuse sessions similar to other model servers.
    """
    claims = None

    if qid:
        session = mwapi.Session(
            get_mediawiki_base_url(protocol), user_agent=custom_ua
        )  # wikidata API session

        try:
            result = session.get(
                action="wbgetentities",
                ids=qid,
                props="claims",
                format="json",
                formatversion=2,
            )
            claims = result.get("entities", {}).get(qid, {}).get("claims", {})
        except Exception as e:
            error_message = (
                f"Failed to get wikibase entities for QID: {qid}. Reason: {e}"
            )
            logging.error(error_message)
            raise InferenceError(error_message)

    return claims


def title_to_categories(
    title: str, lang: str, protocol: str, category_to_country: Dict
) -> Dict:
    """
    Gather categories for an article and check if any map to countries

    TODO: use asynchronous API calls after testing force_http on staging
    Remember to reuse sessions similar to other model servers.
    """
    session = mwapi.Session(
        get_mediawiki_base_url(protocol, lang), user_agent=custom_ua
    )

    try:
        # generate list of all categories for the article and their associated Wikidata IDs
        # https://en.wikipedia.org/w/api.php?action=query&generator=categories&titles=Japanese_iris&prop=pageprops&format=json&ppprop=wikibase_item&gcllimit=max
        result = session.get(
            action="query",
            generator="categories",
            titles=title,
            redirects="",
            prop="pageprops",
            ppprop="wikibase_item",
            gcllimit="max",
            format="json",
            formatversion=2,
        )
        country_categories = {}
        for category in result.get("query", {}).get("pages", []):
            category_qid = category.get("pageprops", {}).get("wikibase_item")
            if category_qid and category_qid in category_to_country:
                country = category_to_country[category_qid]
                category_name = category.get("title")
                country_categories[country] = country_categories.get(country, []) + [
                    category_name
                ]
    except Exception as e:
        error_message = f"Failed to generate list of all categories for the article title: {title}. Reason: {e}"
        logging.error(error_message)
        raise InferenceError(error_message)

    return country_categories


def load_country_properties(file_name: str) -> Dict:
    """
    Load country properties from country_properties.tsv.
    """
    country_properties_tsv = os.path.join(
        current_file_directory, "..", "data", file_name
    )
    country_properties_df = pd.read_csv(country_properties_tsv, sep="\t")
    country_properties = country_properties_df.to_dict()
    logging.debug("Loaded country properties")
    return country_properties


def load_countries_data(file_name: str) -> Dict:
    """
    Load canonical mapping of QID -> region name for labeling from countries.tsv.
    """
    countries_tsv = os.path.join(current_file_directory, "..", "data", file_name)
    countries_df = pd.read_csv(countries_tsv, sep="\t")
    qid_to_region = pd.Series(
        countries_df["name"].values, index=countries_df["wikidata_id"]
    ).to_dict()
    logging.debug(
        f"Loaded {len(qid_to_region)} QID-region pairs for matching against Wikidata -- e.g., Q31: {qid_to_region['Q31']}"
    )
    return qid_to_region


def load_country_aggregations(qid_to_region: Dict, file_name: str) -> Dict:
    """
    Load country-region aggregations from country_aggregation.tsv and update qid_to_region.
    """
    aggregation_tsv = os.path.join(current_file_directory, "..", "data", file_name)
    aggregation_df = pd.read_csv(aggregation_tsv, sep="\t")
    for _, row in aggregation_df.iterrows():
        qid_to = row["QID To"]
        qid_from = row["QID From"]
        if qid_to in qid_to_region:
            qid_to_region[qid_from] = qid_to_region[qid_to]
    logging.debug(
        f"Now {len(qid_to_region)} QID-region pairs after adding aggregations -- e.g., Q40362: {qid_to_region['Q40362']}"
    )
    return qid_to_region


def load_geometries(model_path: str, file_name: str, qid_to_region: Dict) -> Dict:
    """
    Load region geometries from the geojson file and return qid_to_geometry.
    """
    region_geoms_geojson = os.path.join(model_path, file_name)
    qid_to_geometry = {}
    try:
        with open(region_geoms_geojson) as fin:
            regions = json.load(fin)["features"]
            for c in regions:
                qid = c["properties"]["WIKIDATAID"]
                if qid in qid_to_region:
                    qid_to_geometry[qid] = shape(c["geometry"])
                else:
                    logging.debug(
                        f"Skipping geometry for: {c['properties']['NAME']} ({qid})"
                    )
    except Exception as e:
        error_message = (
            f"Failed to load geometries from file: {region_geoms_geojson}. Reason: {e}"
        )
        logging.error(error_message)
        raise InferenceError(error_message)

    for qid in qid_to_region:
        if qid not in qid_to_geometry:
            alt_found = False
            country = qid_to_region[qid]
            for alt_qid in qid_to_region:
                if qid_to_region[alt_qid] == country and alt_qid in qid_to_geometry:
                    alt_found = True
                    break
            if not alt_found:
                logging.debug(f"Missing geometry: {qid_to_region[qid]} ({qid})")
    return qid_to_geometry


def load_categories(model_path: str, file_name: str, qid_to_region: Dict) -> Dict:
    """
    Load category data from category-countries.tsv.gz and return category_to_country dictionary.
    """
    categories_tsv = os.path.join(model_path, file_name)
    category_to_country = {}
    try:
        df_categories = pd.read_csv(
            categories_tsv, sep="\t", compression="gzip", header=0
        )
        assert list(df_categories.columns) == [
            "category_qid",
            "country_name",
        ], "Unexpected header format."

        valid_countries = set(qid_to_region.values())
        valid_categories = df_categories[
            df_categories["country_name"].isin(valid_countries)
        ]
        for _, row in valid_categories.iterrows():
            qid, country = row["category_qid"], row["country_name"]
            if not validate_qid(qid):
                logging.debug(f"Malformed category QID: {qid}")
            elif country not in valid_countries:
                logging.debug(f"Invalid category country: <{country}>")
            else:
                category_to_country[qid] = country
    except Exception as e:
        error_message = (
            f"Failed to load categories from file: {categories_tsv}. Reason: {e}"
        )
        logging.error(error_message)
        raise InferenceError(error_message)
    return category_to_country

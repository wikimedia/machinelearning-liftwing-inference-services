import json
import logging
import os
import re
from typing import Optional, Union

import mwapi
import pandas as pd
import sqlitedict
from aiohttp import ClientSession
from kserve.errors import InferenceError
from shapely.geometry import Point, shape

logging.basicConfig(level=logging.DEBUG)

current_file_directory = os.path.dirname(__file__)
custom_user_agent = "WMF ML Team article-country model inference (LiftWing)"


def init_groundtruth_db(
    model_path: str,
    file_name: str,
) -> Union[sqlitedict.SqliteDict, None]:
    """
    Initialize the groundtruth database (a pre-computed SQLite database using sqlitedict)
    that maps Wikidata QIDs to country strings.
    """
    db_path = os.path.join(model_path, file_name)
    if os.path.exists(db_path):
        try:
            groundtruth_db = sqlitedict.SqliteDict(db_path, autocommit=False)
            logging.info(
                f"Loaded groundtruth SQLite DB with {len(groundtruth_db)} entries."
            )
        except Exception as e:
            logging.error(f"Failed to load groundtruth SQLite DB: {e}")
            groundtruth_db = None
    else:
        logging.warning(f"Groundtruth SQLite DB not found at {db_path}")
        groundtruth_db = None
    return groundtruth_db


def get_groundtruth(qid: str, groundtruth_db: sqlitedict.SqliteDict) -> list[str]:
    """
    For a given QID, return the list of countries from the pre-computed groundtruth DB.
    The stored value is expected to be a ";"-separated string.
    """
    if groundtruth_db:
        record = groundtruth_db.get(qid, "")
        return [c for c in record.split(";") if c]
    return []


async def title_to_links(
    title: str,
    lang: str,
    protocol: str,
    mwapi_client_session: ClientSession,
    groundtruth_db: sqlitedict.SqliteDict,
    limit: int = 500,
) -> dict[str, int]:
    """
    Gather a count of wikilink-based country hints for the given article title.
    For each outlink in the article, if the outlink's Wikidata QID is in the groundtruth DB,
    then count its associated country (or count the absence as an empty string).
    """
    session = mwapi.AsyncSession(
        host=get_mediawiki_base_url(protocol, lang),
        user_agent=custom_user_agent,
        session=mwapi_client_session,
    )
    pages = []
    try:
        # the call with continuation=True returns an async generator
        result = await session.get(
            action="query",
            generator="links",
            titles=title,
            redirects="",
            prop="pageprops",
            ppprop="wikibase_item",
            gplnamespace=0,
            gpllimit=50,
            format="json",
            formatversion=2,
            continuation=True,
        )
        # iterate over the batches yielded by the async result generator
        async for batch in result:
            batch_pages = batch.get("query", {}).get("pages", [])
            pages.extend(batch_pages)
            if len(pages) >= limit:
                break
    except Exception as e:
        logging.error(f"Failed to retrieve links for title {title}: {e}")
        return {}

    country_counts = {}
    processed = 0
    for link in pages:
        processed += 1
        if (
            link.get("ns") == 0 and "missing" not in link
        ):  # namespace 0 and not a red link
            qid = link.get("pageprops", {}).get("wikibase_item")
            if qid:
                link_countries = get_groundtruth(qid, groundtruth_db)
                if link_countries:
                    for entry in link_countries:
                        # calculate cumulative country weights. see P73436#294761
                        country, count = entry.split(":")
                        country_counts[country] = country_counts.get(
                            country, 0
                        ) + float(count)
                else:
                    country_counts[""] = country_counts.get("", 0) + 1
        if processed >= limit:
            break
    return country_counts


def load_country_IDFs(file_name: str) -> dict:
    """
    Load country IDFs from country_IDFs.tsv
    """
    country_IDFs_tsv = os.path.join(os.path.dirname(__file__), "..", "data", file_name)
    country_IDFs_df = pd.read_csv(country_IDFs_tsv, sep="\t", keep_default_na=False)
    country_IDFs = country_IDFs_df.set_index("Country")["IDF"].to_dict()
    logging.debug("Loaded country IDFs")
    return country_IDFs


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


async def title_to_qid(
    lang: str,
    title: str,
    protocol: str,
    mwapi_client_session: ClientSession,
) -> Optional[str]:
    """
    Get Wikidata item ID(s) for Wikipedia article(s)
    """
    session = mwapi.AsyncSession(
        host=get_mediawiki_base_url(protocol, lang),
        user_agent=custom_user_agent,
        session=mwapi_client_session,
    )

    try:
        result = await session.get(
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
    claims: dict, country_properties: dict, qid_to_region: dict
) -> list[tuple]:
    """
    Extracts cultural regions/countries from the provided Wikidata claims.
    """
    regions = []
    for prop in country_properties:
        if prop in claims:
            for statement in claims[prop]:
                try:
                    mainsnak = statement.get("mainsnak")
                    datavalue = mainsnak.get("datavalue")
                    value = datavalue.get("value")
                    entity_id = value.get("id")
                    if entity_id and entity_id in qid_to_region:
                        regions.append((prop, qid_to_region[entity_id]))
                except Exception as e:
                    error_message = f"Failed to get cultural regions for property {prop}. Reason: {e}"
                    logging.error(error_message)
                    # continue to the next statement for processing (see details in T387547)
                    continue
    return regions


def get_geographic_country(
    claims: dict, qid_to_geometry: dict, qid_to_region: dict
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
    lon: float, lat: float, qid_to_geometry: dict, qid_to_region: dict
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


async def get_claims(
    protocol: str, mwapi_client_session: ClientSession, qid: Optional[str] = None
) -> dict:
    """
    Get claims from wikibase entity item of provided QID.
    """
    claims = {}

    if qid:
        session = mwapi.AsyncSession(
            host=get_mediawiki_base_url(protocol),
            user_agent=custom_user_agent,
            session=mwapi_client_session,
        )  # wikidata API session

        try:
            result = await session.get(
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


async def title_to_categories(
    title: str,
    lang: str,
    protocol: str,
    category_to_country: dict,
    mwapi_client_session: ClientSession,
) -> dict:
    """
    Gather categories for an article and check if any map to countries
    """
    session = mwapi.AsyncSession(
        host=get_mediawiki_base_url(protocol, lang),
        user_agent=custom_user_agent,
        session=mwapi_client_session,
    )

    try:
        # generate list of all categories for the article and their associated Wikidata IDs
        # https://en.wikipedia.org/w/api.php?action=query&generator=categories&titles=Japanese_iris&prop=pageprops&format=json&ppprop=wikibase_item&gcllimit=max
        result = await session.get(
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


def load_country_properties(file_name: str) -> dict:
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


def load_countries_data(file_name: str) -> dict:
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


def load_country_aggregations(qid_to_region: dict, file_name: str) -> dict:
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


def load_geometries(model_path: str, file_name: str, qid_to_region: dict) -> dict:
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


def load_categories(model_path: str, file_name: str, qid_to_region: dict) -> dict:
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


def calculate_sums(prediction_data: dict) -> list[float]:
    """
    Calculate the sum of categories, wikidata properties, and wikilinks for each prediction result
    """
    results = prediction_data["prediction"]["results"]
    sums = []
    for result in results:
        num_categories = len(result["source"]["categories"])
        num_wikidata_properties = len(result["source"]["wikidata_properties"])
        # for links, assume there is either 0 or 1 link entry.
        link_signal = 0.0
        links = result["source"].get("links", [])
        if links:
            link_signal = links[0].get("prop-tfidf", 0)
        sums.append(num_categories + num_wikidata_properties + link_signal)
    return sums


def normalize_sums(sums: list[float]) -> list[float]:
    """
    Apply min-max normalization to sums using fixed bounds of 0.0 (min) and 1.0 (max).

    Examples based on P73436#296960:
    1. normalize_sums([4, 1, 2]) returns [1.0, 0.25, 0.5]
    2. normalize_sums([1, 2]) returns [0.5, 1.0]
    3. normalize_sums([0.3, 0.4]) returns [0.3, 0.4]
    4. normalize_sums([0.25, 1]) returns [0.25, 1.0]
    5. normalize_sums([0.25, 10]) returns [0.025, 1.0]

    NB: using 0.25 (min) and 1.0 (max) causes both normalize_sums([0.25, 1])
        and normalize_sums([0.25, 10]) to return [0.25, 1.0]
    """
    min_val = min(sums + [0.0])
    max_val = max(sums + [1.0])
    return [(sum_item - min_val) / (max_val - min_val) for sum_item in sums]


def update_scores(prediction_data: dict, normalized_scores: list[float]) -> None:
    """
    Update scores in the JSON prediction data based on the normalized values
    """
    results = prediction_data["prediction"]["results"]
    for i, result in enumerate(results):
        result["score"] = normalized_scores[i]


def sort_results_by_score(prediction_data: dict) -> dict:
    """
    Sort results in the prediction data so they are ranked by score in descending order
    """
    if "prediction" in prediction_data and "results" in prediction_data["prediction"]:
        prediction_data["prediction"]["results"] = sorted(
            prediction_data["prediction"]["results"],
            key=lambda country_result: country_result["score"],
            reverse=True,
        )
    return prediction_data

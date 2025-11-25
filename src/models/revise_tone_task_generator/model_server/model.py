import datetime
import logging
import os
import re
from http import HTTPStatus
from typing import Any

import aiohttp
import kserve
import mwapi
import mwedittypes.utils
import mwparserfromhell
import torch
from fastapi import HTTPException
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Pipeline,
    pipeline,
)

from python import events
from python.preprocess_utils import (
    get_lang,
    get_page_id,
    get_page_title,
    get_rev_id,
    is_domain_wikipedia,
    validate_json_input,
)

logging.basicConfig(level=kserve.constants.KSERVE_LOGLEVEL)

# Sections to skip per language
SECTIONS_TO_SKIP = {
    "en": [
        "See also",
        "References",
        "External links",
        "Further reading",
        "Notes",
        "Additional sources",
        "Sources",
        "Bibliography",
    ],
    "fr": [
        "Notes et références",
        "Annexes",
        "Bibliographie",
        "Articles connexes",
        "Liens externes",
        "Voir aussi",
        "Notes",
        "Références",
    ],
    "ar": [
        "وصلات خارجية",
        "قراءة موسَّعة",
        "الهوامش",
        "انظر أيضاً",
        "الاستشهاد بالمصادر",
        "انظر أيضًا",
        "مراجع",
    ],
    "pt": [
        "Ver também",
        "Notas e referências",
        "Ligações externas",
        "Referências",
        "Bibliografia",
        "Notas",
    ],
}

# Prefixes for links/files to remove
PREFIXES_TO_REMOVE = {
    "en": ("file:", "image:", "category:"),
    "fr": ("fichier:", "image:", "catégorie:"),
    "ar": ("صورة", "ملف", "تصنيف"),
    "pt": ("file:", "imagem:", "categoria:"),
}

# Model constants
BATCH_SIZE = 200  # Batch size for the model pipeline
MAXLEN = 512  # Maximum length for tokenization

# Article topics to filter for
ALLOWED_TOPICS = {
    "Culture.Biography.Biography*",
    "Culture.Biography.Women",
    "Culture.Sports",
}

TONE_CHECK_TRUE_LABEL = "LABEL_1"


class ReviseToneTaskGenerator(kserve.Model):
    def __init__(self, name: str, use_cache: bool = True) -> None:
        super().__init__(name)
        self.name = name
        self.ready = False
        self.model_path = os.environ.get("MODEL_PATH", "/mnt/models/")
        self.model_version = os.environ.get("MODEL_VERSION", "v1.0")
        self.outlink_topic_model_url = os.environ.get(
            "OUTLINK_TOPIC_MODEL_URL",
            "http://outlink-topic-model.articletopic-outlink:8080/v1/models/outlink-topic-model:predict",
        )
        self.outlink_topic_model_header = os.environ.get("OUTLINK_TOPIC_MODEL_HEADER")

        # Event key for wrapped events
        self.EVENT_KEY = "event"

        # EventGate configuration
        self.EVENTGATE_URL = os.environ.get("EVENTGATE_URL")
        self.EVENTGATE_WEIGHTED_TAGS_CHANGE_STREAM = os.environ.get(
            "EVENTGATE_WEIGHTED_TAGS_CHANGE_STREAM"
        )

        # MediaWiki API configuration
        self.WIKI_URL = os.environ.get("WIKI_URL")
        self.TLS_CERT_BUNDLE_PATH = "/etc/ssl/certs/wmf-ca-certificates.crt"
        self.CUSTOM_UA = "WMF ML Team revise-tone-task-generator svc"
        self.AIOHTTP_CLIENT_TIMEOUT = os.environ.get("AIOHTTP_CLIENT_TIMEOUT", 5)
        self._http_client_session = {}

        self.model_pipeline = self.load()

        self.use_cache = use_cache
        if self.use_cache:
            from model_cache import ReviseToneCache

            self.cache = ReviseToneCache()
            logging.info("Cache initialised!")

    def load(self) -> Pipeline:
        """Load model and resources.

        Returns:
            Pipeline: Transformers pipeline for text classification
        """
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

        # Load the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            truncation=True,
            max_length=MAXLEN,
            padding=True,
            return_tensors="pt",
        )

        # Load the pretrained model
        model = AutoModelForSequenceClassification.from_pretrained(self.model_path)

        # Build the pipeline
        model_pipeline = pipeline(
            task="text-classification",
            model=model,
            tokenizer=tokenizer,
            device=device,
            batch_size=BATCH_SIZE,
        )

        self.ready = True
        logging.info(f"{self.name} model loaded successfully")
        return model_pipeline

    def get_http_client_session(self, endpoint):
        """Returns a aiohttp session for the specific endpoint passed as input.
        We need to do it since sharing a single session leads to unexpected
        side effects (like sharing headers, most notably the Host one)."""
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

    async def send_weighted_tags_change_event(
        self,
        source_data: dict[str, Any],
        weighted_tags: dict[str, Any],
    ) -> None:
        """
        Send a cirrussearch page_weighted_tags_change event to EventGate, generated
        from the source data and prediction results formatted as weighted_tags.

        Args:
            source_data: The source data, either a page_change event or regular payload
            weighted_tags: The weighted_tags structure containing "set" and/or "clear"
        """
        if self.EVENT_KEY in source_data:
            page_change_event = source_data[self.EVENT_KEY]
            weighted_tags_change_event = events.generate_page_weighted_tags_event(
                page_change_event,
                self.EVENTGATE_WEIGHTED_TAGS_CHANGE_STREAM,
                weighted_tags,
            )
        else:
            weighted_tags_change_event = {
                "$schema": "/mediawiki/cirrussearch/page_weighted_tags_change/1.0.0",
                "dt": datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ"),
                "meta": {
                    "stream": self.EVENTGATE_WEIGHTED_TAGS_CHANGE_STREAM,
                    "domain": f"{source_data['lang']}.wikipedia.org",
                },
                "page": {
                    "namespace_id": 0,
                    "page_id": source_data["page_id"],
                    "page_title": source_data["page_title"],
                },
                "weighted_tags": weighted_tags,
                "wiki_id": f"{source_data['lang']}wiki",
                "rev_based": True,
            }

        await events.send_event(
            weighted_tags_change_event,
            self.EVENTGATE_URL,
            self.TLS_CERT_BUNDLE_PATH,
            self.CUSTOM_UA,
            self.get_http_client_session("eventgate"),
        )

    def should_process_article(self, article_topics: dict[str, Any]) -> bool:
        """Check if article topics match the filter criteria.

        Args:
            article_topics: Article topics from outlink-topic-model

        Returns:
            True if at least one topic matches the allowed topics, False otherwise
        """
        if not article_topics:
            return False

        # Extract topic names from the prediction results
        results = article_topics.get("prediction", {}).get("results", [])
        predicted_topics = {
            result.get("topic") for result in results if result.get("topic")
        }

        # Check if there's at least one match
        has_match = bool(predicted_topics & ALLOWED_TOPICS)

        if has_match:
            matching_topics = predicted_topics & ALLOWED_TOPICS
            logging.info(f"Article matches allowed topics: {matching_topics}")
        else:
            logging.info(
                f"Article does not match allowed topics. Predicted: {predicted_topics}"
            )

        return has_match

    def extract_paragraphs(self, text: str, lang: str) -> list[tuple[str, str]]:
        """Extract paragraphs from wikitext content.

        Args:
            text: Wikitext content to parse
            lang: Language code (e.g., 'en')

        Returns:
            List of tuples containing (section_name, paragraph_text)
        """
        if lang == "test":
            lang = "en"  # set testwiki to use enwiki's parameters for parsing
        paragraphs = []

        wikitext = mwparserfromhell.parse(text)
        for section in wikitext.get_sections(levels=[2], include_lead=True):
            # Get the section name
            headings = section.filter_headings()
            if not headings:
                section_name = "LEAD_SECTION"
            else:
                section_name = headings[0].title.strip()
                for heading in headings:
                    section.remove(heading)  # Remove sub-headings
                if section_name in SECTIONS_TO_SKIP.get(lang, []):
                    continue

            # Data cleaning
            for link in section.filter_wikilinks():
                if (
                    link.title.strip()
                    .lower()
                    .startswith(PREFIXES_TO_REMOVE.get(lang, ()))
                ):
                    try:
                        section.remove(link)
                    except ValueError:
                        continue
            for tbl in section.filter_tags(matches=lambda node: node.tag == "table"):
                try:
                    section.remove(tbl)
                except ValueError:
                    continue
            for tpl in section.filter_templates():
                if tpl.name.strip().lower().startswith("infobox"):
                    try:
                        section.remove(tpl)
                    except ValueError:
                        continue

            # Extract paragraphs (more than one newline in between)
            for paragraph in re.split(r"\n+", str(section)):
                if paragraph.startswith(("*", " |")):
                    continue
                plaintext = mwedittypes.utils.wikitext_to_plaintext(
                    paragraph, lang
                ).strip()
                if plaintext.startswith(("*", "{{", "!")):
                    continue
                if "quote>" in plaintext or "<ref>" in plaintext or "|" in plaintext:
                    continue
                if plaintext and len(plaintext) > 100 and len(plaintext) <= 500:
                    paragraphs.append((section_name, plaintext))

        return paragraphs

    async def get_page_content(
        self, lang: str, page_id: int, revision_id: int = None
    ) -> str:
        """Fetch page content (wikitext) from MediaWiki API.

        Args:
            lang: Language code (e.g., 'en')
            page_id: ID of the page
            revision_id: Optional specific revision ID to fetch

        Returns:
            Wikitext content of the page
        """
        session = mwapi.AsyncSession(
            host=self.WIKI_URL or f"https://{lang}.wikipedia.org",
            user_agent=self.CUSTOM_UA,
            session=self.get_http_client_session("mwapi"),
        )
        session.headers["Host"] = f"{lang}.wikipedia.org"

        query_params = {
            "action": "query",
            "prop": "revisions",
            "rvprop": "content",
            "rvslots": "main",
            "formatversion": "2",
            "format": "json",
        }

        # Use revision_id if provided, otherwise use page_id
        if revision_id:
            query_params["revids"] = revision_id
        else:
            query_params["pageids"] = page_id

        try:
            result = await session.get(**query_params)
            pages = result.get("query", {}).get("pages", [])

            if not pages:
                logging.error(
                    f"No page found for page_id={page_id}, revision_id={revision_id}"
                )
                return ""

            page = pages[0]
            if "missing" in page:
                logging.error(f"Page {page_id} is missing")
                return ""

            revisions = page.get("revisions", [])
            if not revisions:
                logging.error(f"No revisions found for page_id={page_id}")
                return ""

            content = revisions[0].get("slots", {}).get("main", {}).get("content", "")
            logging.info(
                f"Fetched content for page_id={page_id}, revision_id={revision_id}"
            )
            return content

        except Exception as e:
            logging.error(
                f"Error fetching content for page_id={page_id}, revision_id={revision_id}: {e}",
                exc_info=True,
            )
            return ""

    async def get_article_topics(self, lang: str, page_id: int) -> dict[str, Any]:
        """
        Query the outlink article topic model from Wikimedia Lift Wing.

        Args:
            lang: Language code (e.g. 'en')
            page_id: ID of the page whose topics we want

        Returns:
            Dict returned directly by the topic model (may contain probabilities,
            topic names, etc.)
        """
        url = self.outlink_topic_model_url

        headers = {
            "Content-Type": "application/json",
            "User-Agent": (
                "Wikimedia-ReviseToneStructuredTask/1.0 "
                "(https://wikitech.wikimedia.org/wiki/Machine_Learning)"
            ),
        }

        # Only add Host header if specified in environment
        if self.outlink_topic_model_header:
            headers["Host"] = self.outlink_topic_model_header

        payload = {
            "lang": lang,
            "page_id": page_id,
        }

        logging.info(f"Requesting article topics for page_id={page_id} ({lang})")

        session = self.get_http_client_session("outlink-topic-model")
        try:
            async with session.post(url, json=payload, headers=headers) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    logging.error(f"Topic model request failed ({resp.status}): {text}")
                    return {}

                return await resp.json()
        except Exception as e:
            logging.error(f"Error fetching article topics: {e}", exc_info=True)
            return {}

    async def preprocess(
        self, inputs: dict[str, Any], headers: dict[str, str] = None
    ) -> dict[str, Any]:
        # Validate and parse JSON input
        inputs = validate_json_input(inputs)

        # Log whether we're processing an event or a regular payload
        if self.EVENT_KEY in inputs:
            logging.info("Preprocessing event")
        else:
            logging.info("Preprocessing regular payload (debug mode)")

        # Extract data using shared utilities (work for both events and regular payloads)
        lang = get_lang(inputs, self.EVENT_KEY)
        page_id = get_page_id(inputs, self.EVENT_KEY)
        page_title = get_page_title(inputs, self.EVENT_KEY)
        revision_id = get_rev_id(inputs, self.EVENT_KEY)

        # Validate lang is supported language
        supported_lang = [lang for lang in SECTIONS_TO_SKIP.keys()] + ["test"]
        if lang not in supported_lang:
            logging.info(f"Unsupported lang: {lang}.")
            raise HTTPException(
                status_code=HTTPStatus.BAD_REQUEST,
                detail=("Unsupported lang."),
            )

        # Get wiki_id for cache operations
        if self.EVENT_KEY in inputs:
            event = inputs[self.EVENT_KEY]
            wiki_id = event.get("wiki_id") or event.get("database")

            # Validate domain is Wikipedia
            if not is_domain_wikipedia(event):
                logging.info("Unsupported domain.")
                raise HTTPException(
                    status_code=HTTPStatus.BAD_REQUEST,
                    detail=(
                        "This model is not recommended for use in projects outside of Wikipedia"
                        " — e.g. Wiktionary, Wikinews, etc."
                    ),
                )
        else:
            wiki_id = f"{lang}wiki"

        # Fetch page content from MW API
        content_body = await self.get_page_content(
            lang=lang, page_id=page_id, revision_id=revision_id
        )

        # Remove old cached predictions for this page before processing
        if self.use_cache and wiki_id and page_id:
            try:
                self.cache.remove_from_cache(wiki_id=wiki_id, page_id=page_id)
            except Exception as e:
                logging.error(f"Failed to remove old cache entries: {e}", exc_info=True)

        paragraphs = self.extract_paragraphs(content_body, lang)

        article_topics = await self.get_article_topics(lang, page_id)

        # Check if article should be processed based on topics
        should_process = self.should_process_article(article_topics)

        preprocessed = {
            "paragraphs": paragraphs,
            "page_id": page_id,
            "page_title": page_title,
            "wiki_id": wiki_id,
            "revision_id": revision_id,
            "lang": lang,
            "article_topics": article_topics,
            "should_process": should_process,
        }

        # Store the event if we're processing an event payload
        if self.EVENT_KEY in inputs:
            preprocessed[self.EVENT_KEY] = inputs[self.EVENT_KEY]

        logging.info(
            f"Extracted {len(paragraphs)} paragraphs; retrieved topics for page_id={page_id}; should_process_topic={should_process}"
        )

        return preprocessed

    async def predict(
        self, request: dict[str, Any], headers: dict[str, str] = None
    ) -> dict[str, Any]:
        """Run inference on the preprocessed request.

        Args:
            request: Preprocessed request dict containing paragraphs, lang, etc.
            headers: Request headers

        Returns:
            Dict containing predictions and original request data
        """
        logging.info("Running inference")

        should_process = request.get("should_process", False)
        paragraphs = request.get("paragraphs", [])
        lang = request.get("lang", "en")

        # Skip prediction if article doesn't match topic criteria
        if not should_process:
            logging.info("Skipping prediction - article does not match topic criteria")
            return {
                "predictions": [],
                "request_data": request,
            }

        if not paragraphs:
            logging.warning("No paragraphs to predict on")
            return {
                "predictions": [],
                "request_data": request,
            }

        # Format paragraphs as model inputs: "lang[SEP]paragraph_text"
        model_inputs = [f"{lang}[SEP]{paragraph[1]}" for paragraph in paragraphs]

        logging.info(f"Running prediction on {len(model_inputs)} paragraphs")

        # Run predictions
        tokenizer_kwargs = {"truncation": True, "max_length": MAXLEN}
        predictions = self.model_pipeline(
            model_inputs, **tokenizer_kwargs, batch_size=BATCH_SIZE
        )

        logging.info(f"Completed prediction for {len(predictions)} paragraphs")

        return {
            "predictions": predictions,
            "request_data": request,
        }

    async def postprocess(
        self, predictions: dict[str, Any], headers: dict[str, str] = None
    ) -> dict[str, Any]:
        """Postprocess the model predictions.

        Args:
            predictions: Dict containing predictions list and request_data
            headers: Request headers

        Returns:
            Formatted response payload
        """
        logging.info("Postprocessing predictions")

        prediction_results = predictions.get("predictions", [])
        request_data = predictions.get("request_data", {})
        paragraphs = request_data.get("paragraphs", [])

        # Filter out negative predictions
        formatted_predictions = []
        for i, (pred, (section_name, text)) in enumerate(
            zip(prediction_results, paragraphs)
        ):
            if pred.get("label") == TONE_CHECK_TRUE_LABEL:
                # The 'paragraph_index' field is reserved for future use when we switch
                # to HTML source. At that point, Growth can use it to locate specific
                # paragraphs in articles instead of relying on string matching.
                formatted_predictions.append(
                    {
                        "paragraph_index": i,
                        "section_name": section_name,
                        "text": text,
                        "score": pred.get("score"),
                    }
                )

        if not formatted_predictions:
            logging.info("No paragraphs with tone issues were found!")

        # Cache the predictions if caching is enabled and we have predictions
        if self.use_cache and formatted_predictions:
            wiki_id = request_data.get("wiki_id")
            page_id = request_data.get("page_id")
            revision_id = request_data.get("revision_id")

            if wiki_id and page_id and revision_id:
                try:
                    self.cache.to_cache(
                        wiki_id=wiki_id,
                        page_id=page_id,
                        revision_id=revision_id,
                        model_version=self.model_version,
                        predictions=formatted_predictions,
                    )
                except Exception as e:
                    logging.error(f"Failed to cache predictions: {e}")

        # Send weighted tags change event after caching
        if self.EVENTGATE_URL:
            if formatted_predictions:
                # Send set event if predictions exist
                # Use the maximum score from all predictions
                max_score = max(pred["score"] for pred in formatted_predictions)
                weighted_tags = {
                    "set": {
                        "recommendation.tone": [{"tag": "exists", "score": max_score}]
                    }
                }
                try:
                    await self.send_weighted_tags_change_event(
                        request_data, weighted_tags
                    )
                    logging.info(
                        f"Sent set weighted tag event for page_id={request_data.get('page_id')} "
                        f"with max_score={max_score:.4f}"
                    )
                except Exception as e:
                    logging.error(
                        f"Failed to send set weighted tags event: {e}", exc_info=True
                    )
            else:
                # Send clear event if no predictions exist
                weighted_tags = {"clear": ["recommendation.tone"]}
                try:
                    await self.send_weighted_tags_change_event(
                        request_data, weighted_tags
                    )
                    logging.info(
                        f"Sent clear weighted tag event for page_id={request_data.get('page_id')}"
                    )
                except Exception as e:
                    logging.error(
                        f"Failed to send clear weighted tags event: {e}", exc_info=True
                    )

        return {
            "page_id": request_data.get("page_id"),
            "page_title": request_data.get("page_title"),
            "wiki_id": request_data.get("wiki_id"),
            "revision_id": request_data.get("revision_id"),
            "lang": request_data.get("lang"),
            "article_topics": request_data.get("article_topics"),
            "should_process": request_data.get("should_process"),
            "predictions": formatted_predictions,
        }


if __name__ == "__main__":
    use_cache = os.environ.get("USE_CACHE", "false").lower() == "true"
    model = ReviseToneTaskGenerator(
        name="revise-tone-task-generator", use_cache=use_cache
    )
    kserve.ModelServer(workers=1).start([model])

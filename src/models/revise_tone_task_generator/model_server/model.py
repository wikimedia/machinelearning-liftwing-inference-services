import logging
import os
import re
from typing import Any

# import aiohttp
import kserve
import mwedittypes.utils
import mwparserfromhell
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Pipeline,
    pipeline,
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
}

# Prefixes for links/files to remove
PREFIXES_TO_REMOVE = ("file:", "image:", "category:")

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
        # self.outlink_topic_model_url = os.environ.get(
        #     "OUTLINK_TOPIC_MODEL_URL",
        #     "http://outlink-topic-model.articletopic-outlink:8080/v1/models/outlink-topic-model:predict",
        # )
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

    def should_process_article(self, article_topics: dict[str, Any]) -> bool:
        """Check if article topics match the filter criteria.

        Args:
            article_topics: Article topics from outlink-topic-model

        Returns:
            True if at least one topic matches the allowed topics, False otherwise
        """
        # Temporarily disabled - always return True to process all articles
        return True
        # if not article_topics:
        #     return False
        #
        # # Extract topic names from the prediction results
        # results = article_topics.get("prediction", {}).get("results", [])
        # predicted_topics = {
        #     result.get("topic") for result in results if result.get("topic")
        # }
        #
        # # Check if there's at least one match
        # has_match = bool(predicted_topics & ALLOWED_TOPICS)
        #
        # if has_match:
        #     matching_topics = predicted_topics & ALLOWED_TOPICS
        #     logging.info(f"Article matches allowed topics: {matching_topics}")
        # else:
        #     logging.info(
        #         f"Article does not match allowed topics. Predicted: {predicted_topics}"
        #     )
        #
        # return has_match

    def extract_paragraphs(self, text: str, lang: str) -> list[tuple[str, str]]:
        """Extract paragraphs from wikitext content.

        Args:
            text: Wikitext content to parse
            lang: Language code (e.g., 'en')

        Returns:
            List of tuples containing (section_name, paragraph_text)
        """
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
                if link.title.strip().lower().startswith(PREFIXES_TO_REMOVE):
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
                if paragraph.startswith("*"):  # Remove bullet points
                    continue
                if paragraph.startswith(" |"):  # Remove table leftover
                    continue
                plaintext = mwedittypes.utils.wikitext_to_plaintext(
                    paragraph, lang
                ).strip()
                if plaintext and len(plaintext) > 30:  # Paragraphs more than 30 chars
                    paragraphs.append((section_name, plaintext))

        return paragraphs

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
        # Temporarily disabled - return empty dict
        return {}
        # url = self.outlink_topic_model_url
        #
        # headers = {
        #     "Content-Type": "application/json",
        #     "User-Agent": (
        #         "Wikimedia-ReviseToneStructuredTask/1.0 "
        #         "(https://wikitech.wikimedia.org/wiki/Machine_Learning)"
        #     ),
        # }
        #
        # payload = {
        #     "lang": lang,
        #     "page_id": page_id,
        # }
        #
        # logging.info(f"Requesting article topics for page_id={page_id} ({lang})")
        #
        # async with aiohttp.ClientSession() as session:
        #     async with session.post(url, json=payload, headers=headers) as resp:
        #         if resp.status != 200:
        #             text = await resp.text()
        #             logging.error(f"Topic model request failed ({resp.status}): {text}")
        #             return {}
        #
        #         return await resp.json()

    async def preprocess(
        self, inputs: dict[str, Any], headers: dict[str, str] = None
    ) -> dict[str, Any]:
        logging.info("Preprocessing mediawiki.page_content_change.v1 event")

        revision_info = inputs.get("revision", {})
        content_body = (
            revision_info.get("content_slots", {})
            .get("main", {})
            .get("content_body", "")
        )
        revision_id = revision_info.get("rev_id")

        page_info = inputs.get("page", {})
        page_id = page_info.get("page_id")
        page_title = page_info.get("page_title")
        wiki_id = inputs.get("wiki_id")

        lang = wiki_id.replace("wiki", "") if wiki_id else "en"

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

        logging.info(
            f"Extracted {len(paragraphs)} paragraphs; retrieved topics for page_id={page_id}; should_process={should_process}"
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

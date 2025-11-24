import logging
from typing import Any

from cassandra import ConsistencyLevel
from cassandra.cqlengine import columns
from cassandra.cqlengine.models import Model

from python.cassandra_cache import BaseCassandraCache, CassandraSettings

# Get logger that will inherit kserve's logging configuration
logger = logging.getLogger(__name__)


class PageParagraphToneScore(Model):
    """
    Cassandra model for caching paragraph tone scores.

    Stores tone classification scores for individual paragraphs within Wikipedia pages.
    The partition key (wiki_id, page_id) groups all paragraphs from the same page,
    while clustering keys (revision_id, model_version, idx) provide versioning
    and ordering within each page.

    Table Schema:
        Partition Key: (wiki_id, page_id)
        Clustering Keys: revision_id, model_version, idx
    """

    # Partition key
    wiki_id = columns.Text(partition_key=True, required=True)
    page_id = columns.BigInt(partition_key=True, required=True)

    # Clustering keys
    revision_id = columns.BigInt(
        primary_key=True, required=True, clustering_order="ASC"
    )
    model_version = columns.Text(
        primary_key=True, required=True, clustering_order="ASC"
    )
    idx = columns.Integer(primary_key=True, required=True, clustering_order="ASC")

    # Data columns
    content = columns.Text(required=True)
    score = columns.Float(required=True)


class ReviseToneCache(BaseCassandraCache):
    """
    Cache implementation for Revise Tone Task Generator predictions.

    Manages caching of paragraph-level tone scores for Wikipedia pages. The cache
    helps avoid redundant model inference by storing results per revision, allowing
    quick retrieval of previously computed scores.

    The cache is keyed by:
        - wiki_id: Wikipedia database identifier (e.g., "enwiki")
        - page_id: Numeric page identifier
        - revision_id: Revision identifier for versioning
        - model_version: Model version used for predictions
        - idx: Paragraph index within the page

    Usage:
        settings = CassandraSettings(cassandra_keyspace="ml_cache")
        cache = ReviseToneCache(settings=settings)

        # Check cache before inference
        cached_data = cache.from_cache(
            wiki_id="enwiki", page_id=12345, revision_id=67890, model_version="v1.0"
        )
        if cached_data:
            predictions = cached_data["predictions"]
        else:
            # Run inference...
            cache.to_cache(
                wiki_id="enwiki", page_id=12345, revision_id=67890,
                model_version="v1.0", predictions=results
            )
    """

    def __init__(
        self,
        settings: CassandraSettings | None = None,
    ) -> None:
        """
        Initialize the Revise Tone cache.

        Args:
            model_version: Version identifier for the model (e.g., "v1.0")
            settings: Cassandra settings. If None, will be loaded from environment.
        """
        super().__init__(
            cache_model_class=PageParagraphToneScore,
            settings=settings,
        )

    def from_cache(
        self,
        wiki_id: str,
        page_id: int,
        revision_id: int,
        model_version: str,
    ) -> dict[str, Any] | None:
        """
        Retrieve cached predictions for a specific page revision.

        Args:
            wiki_id: Wikipedia database identifier (e.g., "enwiki")
            page_id: Numeric page identifier
            revision_id: Revision identifier
            model_version: Model version identifier

        Returns:
            Dict containing cached predictions in the format:
            {
                "predictions": [
                    {
                        "paragraph_index": 0,
                        "content": "paragraph text",
                        "score": 0.95
                    },
                    ...
                ]
            }
            Returns None if no cache entry exists for this revision.
        """
        try:
            # Query all paragraphs for this page revision
            results = PageParagraphToneScore.objects.filter(
                wiki_id=wiki_id,
                page_id=page_id,
                revision_id=revision_id,
                model_version=model_version,
            ).all()

            if not results:
                logger.info(
                    f"Cache miss: wiki_id={wiki_id}, page_id={page_id}, "
                    f"revision_id={revision_id}, model_version={model_version}"
                )
                return None

            # Convert to prediction format
            predictions = [
                {
                    "paragraph_index": result.idx,
                    "content": result.content,
                    "score": result.score,
                }
                for result in results
            ]

            # Sort by paragraph index to maintain order
            predictions.sort(key=lambda x: x["paragraph_index"])

            logger.info(
                f"Cache hit: wiki_id={wiki_id}, page_id={page_id}, "
                f"revision_id={revision_id}, found {len(predictions)} paragraphs"
            )

            return {"predictions": predictions}

        except Exception as e:
            logger.error(f"Error retrieving from cache: {e}")
            return None

    def to_cache(
        self,
        wiki_id: str,
        page_id: int,
        revision_id: int,
        model_version: str,
        predictions: list[dict[str, Any]],
    ) -> None:
        """
        Store predictions in cache for a specific page revision.

        Args:
            wiki_id: Wikipedia database identifier (e.g., "enwiki")
            page_id: Numeric page identifier
            revision_id: Revision identifier
            model_version: Model version identifier
            predictions: List of prediction dicts, each containing:
                - paragraph_index: Index of the paragraph (required)
                - text or content: Paragraph text (required)
                - score: Tone classification score (required)

        Raises:
            Exception: If caching fails (logged but not re-raised)
        """
        try:
            cached_count = 0
            for pred in predictions:
                # Extract required fields
                idx = pred.get("paragraph_index")
                content = pred.get("text") or pred.get("content")
                score = pred.get("score")

                # Skip invalid entries
                if idx is None or content is None or score is None:
                    logger.warning(f"Skipping invalid prediction entry: {pred}")
                    continue

                # Create cache entry with TTL and LOCAL_QUORUM consistency
                PageParagraphToneScore.ttl(self.ttl).consistency(
                    ConsistencyLevel.LOCAL_QUORUM
                ).create(
                    wiki_id=wiki_id,
                    page_id=page_id,
                    revision_id=revision_id,
                    model_version=model_version,
                    idx=idx,
                    content=content,
                    score=score,
                )
                cached_count += 1

            logger.info(
                f"Cached {cached_count} predictions: wiki_id={wiki_id}, "
                f"page_id={page_id}, revision_id={revision_id}, "
                f"model_version={model_version}"
            )

        except Exception as e:
            logger.error(
                f"Error writing to cache for page_id={page_id}, "
                f"revision_id={revision_id}: {e}"
            )

    def remove_from_cache(
        self,
        wiki_id: str,
        page_id: int,
        revision_id: int | None = None,
        model_version: str | None = None,
    ) -> None:
        """
        Remove cached predictions for a specific page.

        Can remove all predictions for a page, or filter by specific revision_id
        and/or model_version.

        Args:
            wiki_id: Wikipedia database identifier (e.g., "enwiki")
            page_id: Numeric page identifier
            revision_id: Optional revision identifier to filter deletion
            model_version: Optional model version identifier to filter deletion

        Examples:
            # Remove all predictions for a page
            cache.remove_from_cache(wiki_id="enwiki", page_id=12345)

            # Remove predictions for specific revision
            cache.remove_from_cache(
                wiki_id="enwiki", page_id=12345, revision_id=67890
            )

            # Remove predictions for specific model version
            cache.remove_from_cache(
                wiki_id="enwiki", page_id=12345, model_version="v1.0"
            )

            # Remove predictions for specific revision and model version
            cache.remove_from_cache(
                wiki_id="enwiki", page_id=12345,
                revision_id=67890, model_version="v1.0"
            )
        """
        try:
            # Build the query based on provided filters
            query = PageParagraphToneScore.objects.filter(
                wiki_id=wiki_id,
                page_id=page_id,
            )

            if revision_id is not None:
                query = query.filter(revision_id=revision_id)

            if model_version is not None:
                query = query.filter(model_version=model_version)

            # Delete using the query's delete method
            query.delete()

            logger.info(
                f"Removed cache entries for wiki_id={wiki_id}, page_id={page_id}"
            )

        except Exception as e:
            logger.error(
                f"[CACHE] Error removing from cache for wiki_id={wiki_id}, "
                f"page_id={page_id}: {e}",
                exc_info=True,
            )

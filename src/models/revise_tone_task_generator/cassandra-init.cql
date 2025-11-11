-- Initialize Cassandra keyspace and tables for ML cache

CREATE KEYSPACE IF NOT EXISTS ml_cache
  WITH replication = {'class': 'SimpleStrategy', 'replication_factor': 1};

CREATE TABLE IF NOT EXISTS ml_cache.page_paragraph_tone_scores (
    wiki_id        text,
    page_id        bigint,
    revision_id    bigint,
    model_version  text,
    content        text,
    score          float,
    idx            int,
    PRIMARY KEY ((wiki_id, page_id), revision_id, model_version, idx)
);

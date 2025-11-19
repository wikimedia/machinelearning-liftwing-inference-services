# Revise Tone Task Generator

The Tone Suggestion Generator is a streaming application in Lift Wing that sources page_change events and updates the necessary systems for downstream use. To learn more about the use-case, please visit the (Phabricator page.)[https://phabricator.wikimedia.org/T408538]

1. Consumes events: `mediawiki.page_change.v1`
2. Filters the articles to a selected group of article topics. Topics are retrieved by querying article topic model on LiftWing.
3. Splits the article into paragraphs and runs tone check inference on each paragraph.
4. Paragraphs with tone check issues are saved into Cassandra table.
5. We update the search weighed tags by emitting events: `mediawiki.cirrussearch.page_weighted_tags_change.v1`

## Local Cassandra setup

For local development, Cassandra service was added to the docker compose in `src/models/revise_tone_task_generator/docker-compose.yml`.
There are also 2 initialzation files helping to set up local Cassandra tables:
- **`cassandra-init.cql`**: Schema definition for `ml_cache.page_paragraph_tone_scores` table
- **`cassandra-entrypoint.sh`**: Initialization script that starts Cassandra and applies schema automatically

## Usage

This model has its own `docker-compose.yml` file for easier local development and testing.

### Testing the Model and Cache

To test the model server with a sample request and verify caching:

**1. Start Cassandra and the model server:**
```bash
PATH_TO_REVISE_TONE_TASK_GENERATOR_MODEL=/path/to/tone_check/model docker-compose -f src/models/revise_tone_task_generator/docker-compose.yml up cassandra revise-tone-task-generator-cpu
```

**2. Send a sample prediction request:**
```bash
curl -X POST http://localhost:8080/v1/models/revise-tone-task-generator:predict \
  -H "Content-Type: application/json" \
  -d @test/unit/revise_tone_task_generator/sample_payload.json
```

**3. Verify data was cached in Cassandra:**
```bash
docker exec revise-tone-cassandra cqlsh -e "SELECT wiki_id, page_id, revision_id, model_version, idx, score FROM ml_cache.page_paragraph_tone_scores;"
```

### Running Tests

To run the unit tests with Docker:

```bash
docker-compose -f src/models/revise_tone_task_generator/docker-compose.yml build revise-tone-task-generator-test && docker-compose -f src/models/revise_tone_task_generator/docker-compose.yml run --rm revise-tone-task-generator-test
```

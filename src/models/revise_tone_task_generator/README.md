# Revise Tone Task Generator

The Tone Suggestion Generator is a streaming application in Lift Wing that sources page_content_change events and updates the necessary systems for downstream use. It does the following:

- **Consumes events** (triggered by changeprop): `mediawiki.page_content_change.v1`
- **Filters pages**, processes and parses content into paragraphs, then gets scores from the Tone Check model
- **Updates the search weighted tags** by emitting events: `mediawiki.cirrussearch.page_weighted_tags_change.v1`
- **Updates (inserts/deletes) data in Cassandra**

## Model Structure

The model server implements the standard KServe interface with three main methods:

- `preprocess`: Validates and preprocesses the input request
- `predict`: Runs inference using the loaded model
- `postprocess`: Formats the predictions into the response payload

## Usage

### Running the Model Server

The model server can be run with either CPU or GPU support:

**GPU Version (requires AMD GPU with ROCm support):**
```bash
PATH_TO_REVISE_TONE_TASK_GENERATOR_MODEL=/path/to/tone_check/model docker-compose up revise-tone-task-generator
```

**CPU Version:**
```bash
PATH_TO_REVISE_TONE_TASK_GENERATOR_MODEL=/path/to/tone_check/model docker-compose up revise-tone-task-generator-cpu
```

### Running Tests

To run the unit tests with Docker:

```bash
docker-compose build revise-tone-task-generator-test && docker-compose run --rm revise-tone-task-generator-test
```

## TODO

- [x] Consume and process `mediawiki.page_content_change.v1` events (triggered by changeprop)
- [x] Parse content into paragraphs
- [x] Fetch article topics from outlink-topic-model API
- [x] Get scores from the Tone Check model for each paragraph
- [x] Filter pages based on article topic criteria (Culture.Biography.Biography*, Culture.Biography.Women, Culture.Sports)
- [ ] Emit `mediawiki.cirrussearch.page_weighted_tags_change.v1` events to update search weighted tags
- [ ] Update (insert/delete) data in Cassandra

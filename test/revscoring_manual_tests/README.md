# Liftwing Revscoring API tests
This directory contains API tests that are run manually.
There are two types of tests and both read the `deployed_models.yaml` and the `rev_ids.json` to extract information about the models and the revisions that are used for testing.
Revision ids for each wiki have been extracted from the `event_sanitized.mediawiki_revision_score` table.

The tests assert that given a proper request we get a 200 response and that the word "probability" is in the response's text.
This is sufficient to assert that the revscoring API(s) is working as expected.
However, they should be extended/adapted if used to facilitate all Liftwing use cases.
We recommend running the tests through httpbb as it is more reliable and easier to run.
The second type of tests is kept here more for reference and quick experimentation especially in case something is not supported by httpbb.
Even in this case, after one figures out the type of tests needed, it is recommended to file a patch to add the missing functionality to httpbb.

1. Tests that are run through [httpbb](https://wikitech.wikimedia.org/wiki/Httpbb)

   The script `create_httpb_tests.py` reads the `deployed_models.yaml` and the `rev_ids.json` and creates the configuration yaml files that can be run through httpbb.
   The output of these files is version controlled in the [puppet](https://gerrit.wikimedia.org/r/plugins/gitiles/operations/puppet) repository.

   In order to run the API test suite through httpbb, one has to run the following commands from the deployment server:

   **Staging**:
   ```bash
   httpbb --host inference-staging.svc.codfw.wmnet --https_port 30443 /srv/deployment/httpbb-tests/liftwing/staging/*
   ```

   **Production**:
   ```bash
   httpbb --host inference.svc.codfw.wmnet --https_port 30443 /srv/deployment/httpbb-tests/liftwing/production/*
   ```
   ```bash
   httpbb --host inference.svc.eqiad.wmnet --https_port 30443 /srv/deployment/httpbb-tests/liftwing/production/*
   ```

2. Tests that are run through the python script `endpoint_manual_testing.py` from deployment server

   The script will query the endpoints and will raise an error if all the revision ids for
a model server fail to give a 200 response and do not contain the word probability in the text
of the response (since all revscoring models give a response with probabilities in it).

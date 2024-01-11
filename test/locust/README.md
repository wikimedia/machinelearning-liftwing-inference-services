 _# Load testing with locust
 [![locust](https://img.shields.io/badge/locust-2.20.1-blue.svg)](https://locust.io/)
 [![python](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/)

We are using [locust](https://locust.io/) to do load testing.

The load test class for each model server is defined in a file under /models directory.
All load tests are run from the same locustfile.py script.
We enable a load test for a model server to run just by adding the top level import statement for the load test file in locustfile.py.
e.g.
```python
from models.revertrisk import RevertriskLanguageAgnostic, RevertriskMultilingual  # noqa
```
(Note: the `#noqa` is to suppress the unused import warning, otherwise ruff will erase this line)

The first time we run the load tests for all model servers we run the following command which will produce
a csv file with the results for each model server.
```bash
locust --csv results
```
This will produce the following files: results_stats.csv, results_stats_history.csv,
results_failures.csv, results_exceptions.csv.

We commit only the file `results_stats.csv` to the repo which we will use in subsequent runs to compare the results.

Next time we introduce a change and want to run a load test we run the following command:
```bash
locust
```
### Running a specific load test
Our load tests are defined in an hierarchical structure.
That means that by default all load tests are run.
If we want to run a specific load test we can do it by specifying the environment variable MODEL
to match one of the module names under the model/ directory
For example if we want to run the load tests for revertrisk models we run the following command:
```bash
MODEL=revertrisk locust
```

 # Load testing with locust
 [![locust](https://img.shields.io/badge/locust-2.20.1-blue.svg)](https://locust.io/)
 [![python](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/)

We are using [locust](https://locust.io/) to do load testing.

The load test class for each model server is defined in a file under /models directory.
All load tests are run from the same locustfile.py script.
We enable a load test for a model server to run just by adding the top level import statement for the load test file in
locustfile.py.
In our case we are automatically adding the import statements for all load test files under the model/ directory as long
as they are declared as python modules and their __init__.py has the appropriate import statement.

### Running all load tests
In order to run all load tests we run the following command:
```bash
locust
```
This will provide the load test results in the console and it will also run a comparison with the
previous recorded results (the ones that exist under the results/ directory).


### Running a specific load test
Our load tests are defined in an hierarchical structure.
That means that by default all load tests are run.
If we want to run a specific load test we can do it by specifying the environment variable MODEL
to match one of the module names under the model/ directory
For example if we want to run the load tests for revertrisk models we run the following command:
```bash
MODEL=revertrisk locust
```

### Adding a new test
The first time we introduce a model server we need to do the following:
1. Create a new file under the models directory with the load test class and declare it as a python module
   by adding the appropriate import statement in the __init__.py file. (check the existing models for examples)
2. Run the load test and commit the `***_stats.csv` file under the results directory.

#### Example:
We add a new model named `mycoolmodel` under the models directory. The load test class is defined in the file
`mycoolmodel.py` and is named `MyCoolModelTest`. We add the following import statement in the __init__.py file:
```python
from .mycoolmodel import MyCoolModelTest  # noqa
```
We run the load test and commit the `mycoolmodel_stats.csv` file under the results directory.
```bash
MODEL=mycoolmodel locust --csv results/mycoolmodel
```

In a similar way we produced the results for the existing model servers with the following commands:
```bash
MODEL=revscoring locust --csv results/revscoring
MODEL=revertrisk locust --csv results/revertrisk
```

### Updating results after a change
In the case where we want to update the load test results for a specific server because we made a change in the
model server code that affects load and/or latencies we run the following command:
```bash
MODEL=revertrisk locust --csv results/revertrisk
```
and then commit the updated `revertrisk_stats.csv` file under the results directory.


## Huggingface models

To run a load test for a huggingface model, apart from MODEL we also need to specify the MODEL_NAME and the HOST.
This is done so that we can use the same locust model file for all deployed huggingface models.
Also the NAMESPACE is an optional environment variable that can be used to specify the namespace of the model server.
```bash
MODEL=huggingface MODEL_NAME=gemma2 HOST=gemma2-27b-it locust
```


## Running load tests using the Makefile

### 1. Build
This build process will: set up a Python virtual environment, install dependencies, run locust load tests for the specified model isvc, and update csv results in `results/model_name.csv`.
```bash
MODEL_LOCUST_DIR="logo_detection" make run-locust-test
```

### 2. Remove
If you would like to remove the setup run:
```bash
make clean
```

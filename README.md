# Inference Services

KServe model servers and configs for Lift Wing

## CI - pre-commit

For each of the model servers we use a set of pre-commit hooks to ensure that the code is formatted and linted correctly.
We use the pre-commit framework which is *a framework for managing and maintaining multi-language pre-commit hooks*,
as stated in its [official documentation](https://pre-commit.com).

The pre-commit hooks are defined in the `.pre-commit-config.yaml` file. During CI - which is currently executed through
Jenkins - we execute exactly the same commands as defined in tox.ini.
First install the [pre-commit]() package using pip:
> pip install pre-commit

To install the pre-commit hooks in your local setup run the following command:
> pre-commit install
>
This will enable the hooks described below to run automatically before each commit.
If one wants to bypass them one can use the `--no-verify` or `-n` flag when committing.
Alternatively, one can run the pre-commit hooks manually by running the following command:
> pre-commit run --all-files

We use the following pre-commit hooks:

* check-yaml- checks yaml files for parseable syntax.
* end-of-file-fixer - makes sure files end with a newline
* trailing-whitespace - trims trailing whitespace
* [pyupgrade](https://github.com/asottile/pyupgrade) - upgrade syntax for newer versions of the language
* [black](https://github.com/psf/black) - black is a Python auto code formatter
* [ruff](https://github.com/charliermarsh/ruff) - ruff is a linting tool for Python

A list of the out-of-the-box available pre-commit hooks can be found [here](https://pre-commit.com/hooks.html).

## Running locally
> [!NOTE]
> Please ensure the following prerequisites are installed on your system: Python3, pip, wget, git, cmake, pkg-config, and make.
> `make article-descriptions` works well on linux (statbox and ml-sandbox) and it appears that the model download issue faced on some machines is due to a cap from the analytics public repository website.
To start a model server locally, you can use the `make` command with the appropriate target. For example, running `make revertrisk-language-agnostic` sets up the revertrisk-language-agnostic model server. The [Makefile](Makefile) includes all required commands to download the model from the [public repository](https://analytics.wikimedia.org/published/wmf-ml-models/), create a virtual environment and install the necessary packages for the model.

If the model server is running, you will see something similar to the following:
```
...
2024-01-19 16:04:34.830 uvicorn.error INFO:     Application startup complete.
2024-01-19 16:04:34.831 uvicorn.error INFO:     Uvicorn running on http://0.0.0.0:8080 (Press CTRL+C to quit)
```

Send a test inference request in another terminal:
> curl localhost:8080/v1/models/revertrisk-language-agnostic:predict -i -X POST -d '{"lang": "en", "rev_id": 12345}'

Expected Output:
```
HTTP/1.1 200 OK
date: Fri, 19 Jan 2024 08:03:18 GMT
server: uvicorn
content-length: 206
content-type: application/json

{"model_name":"revertrisk-language-agnostic","model_version":"3","wiki_db":"enwiki","revision_id":12345,"output":{"prediction":false,"probabilities":{"true":0.17687281966209412,"false":0.8231271803379059}}}
```

If you want to keep your workspace clean after testing or developing with model servers, you can use the `make clean` command with the appropriate model server parent directory to easily remove the files generated by the `make` build. For example, if you run `MODEL_SERVER_PARENT_DIR=revert_risk_model make clean`, the build files for the revertrisk-language-agnostic model server will be removed:
```
rm -rf __pycache__
rm -rf my_venv
Cleaning models in directory revert_risk_model ...
```

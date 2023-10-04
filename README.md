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
If you want to run the model servers locally you can do so by first adding the top level dir to the PYTHONPATH
> export PYTHONPATH=$PYTHONPATH:
>
Then running:
>  INFERENCE_NAME=enwiki-damaging MODEL_PATH=/path/to/model.bin WIKI_URL=https://en.wikipedia.org python revscoring_model/model.py

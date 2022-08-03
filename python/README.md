# How is this Python code used?

The `python` directory is used in the Blubber config (see the `.pipeline` directory)
to share code between model server implementations (the `model.py` files).
There is a glob `*.py` in various Blubber configs that copies all Python modules
defined in this directory to the model-server one (created and added in
the Docker images). Please note that there is still no configuration to copy
.py files created in subdirectories. All .py files copied from this directory
will be added in the same directory as the target `model.py` file.

# How are dependencies handled?

The `python` directory contains a `requirements.txt` file that is evaluated
by Blubber when building the Dockerfile. In this way we can list dependencies
needed for the shared code as well.
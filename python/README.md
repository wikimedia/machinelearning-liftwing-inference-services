# How is this Python code used?

The `python` directory is used in the Blubber config (see the `.pipeline` directory)
to share code between model server implementations (the `model.py` files).
There is a glob `*.py` in various Blubber configs that copies all Python modules
defined in this directory to the model-server one (created and added in
the Docker images). Please note that there is still no general configuration to copy
.py files created in subdirectories. All .py files copied from this directory
will be added in the same directory as the target `model.py` file.

There is only one exception to the above rule, namely the `revscoring`
subdirectory. It has its own set of .py modules but not a requirements.txt file.
The idea is to have a single place where all the revscoring-related code is
added, in order to avoid carrying the `revscoring` dependency over to models
that don't come from ORES or that are not revscoring-based.

## Why don't we have a dedicated revscoring requirements.txt file?

Historically solving dependencies for revscoring and related libraries
like `{draft,article,edit,topic}quality` has been particularly challenging
and the time dedicated to it was a lot. Since we are planning to deprecate
the revscoring model-servers, for the moment it seems wiser to keep it simple
and track/duplicate the dependency list on the model-server's requirements.txt
file. We are not going to do it for the new model servers of course, this is
an exception only for revscoring-based ones.

# How are dependencies handled?

The `python` directory contains a `requirements.txt` file that is evaluated
by Blubber when building the Dockerfile. In this way we can list dependencies
needed for the shared code as well.

# What Docker images are rebuilt when a Python module in this directory is changed?

In the [integration/config](https://gerrit.wikimedia.org/r/admin/repos/integration/config)
repository there is a config file called `zuul/layout.yaml`, it contains all
the instructions to rebuild Docker images when specific files are changed/merged.

For example:

```
- name: ^trigger-inference-services-pipeline-editquality
  files:
      - '.pipeline/editquality/blubber.yaml'
      - '.pipeline/config.yaml'
      - 'revscoring/editquality/model-server/.*'
      - 'python/.*\.py$'
```

So any change in this directory's Python files will trigger a complete image
rebuild.

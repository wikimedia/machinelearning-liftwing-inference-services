"""Stubs torch and transformers for every test in this directory.

`qa.py` reads torch while it is imported, to build `DTYPES`, and imports two
classes from transformers. The tests need neither for real: the model is
monkeypatched out, so CI does not pull a multi-hundred-MB wheel to test span
arithmetic and the response envelope. This follows test/unit/qwen36, which stubs
vLLM the same way.

`test_boundaries.py` needs none of this. `boundaries.py` reads no torch and no
transformers, so it imports on any Python. The stubs are here for the modules
that reach `qa.py`: `test_model.py` and `test_highlighter.py`.

`test_real_tokenizer.py` needs the opposite -- the real transformers -- so
`SH_REAL_MODEL=1` turns the stubs off. See that module for how to run it.

The stubs live here rather than in each test module so that there is one copy.

**Two details are deliberate, and both come from other directories mocking the
same modules.** Four test directories mock torch or transformers, and two of
them assign `sys.modules[...]` unconditionally rather than with `setdefault`.

1. The stubs are installed from a hook, not only when this file is imported.
   pytest loads every conftest at startup, before it imports any test module,
   so a stub installed at import time is in place too early to survive:
   `test/unit/embeddings/test_model_transformers.py` replaces
   `sys.modules["torch"]` when *its* module is imported, which is later.
   `pytest_collectstart` runs immediately before pytest imports each module
   here, which is after every other directory has had its say.

2. A module another directory installed is filled in, not replaced. Rather than
   fight over it, this asks only for the names `qa.py` reads and adds the ones
   that are absent. `embeddings` installs a bare module carrying `bfloat16` and
   `float16` but no `float32`, and sorts before `semantic_highlighting`, so
   under `pytest test/unit` every test file here used to fail to import with
   `AttributeError: module 'torch' has no attribute 'float32'`. Whatever the
   other directory set, it keeps.
"""

import os
import sys
from unittest.mock import MagicMock

# `test_real_tokenizer.py` runs against the tokenizer the service really loads,
# so it needs the real transformers rather than a stub. It is skipped unless
# this is set, and when it is set nothing here installs a stub: an image that
# carries torch and transformers keeps them, and one that does not still runs
# every other module here because those import neither at module level.
_REAL = os.environ.get("SH_REAL_MODEL") == "1"

# What qa.py reads from each module while it is being imported. Anything the
# stubs are missing is added; nothing already there is replaced.
_REQUIRED = {
    "torch": ("float32", "bfloat16", "float16"),
    "transformers": ("AutoModelForQuestionAnswering", "AutoTokenizer"),
}


def _stub_package(name):
    """A stand-in module that answers any attribute, and `from x import y`."""
    module = MagicMock()
    module.__path__ = []
    module.__name__ = name
    return module


def _install_stubs():
    if _REAL:
        return
    for name, attributes in _REQUIRED.items():
        module = sys.modules.setdefault(name, _stub_package(name))
        for attribute in attributes:
            if not hasattr(module, attribute):
                setattr(module, attribute, MagicMock(name=f"{name}.{attribute}"))


def pytest_collectstart(collector):
    """Install the stubs just before pytest imports a module in this directory."""
    _install_stubs()


_install_stubs()

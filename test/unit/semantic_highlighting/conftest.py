"""Stubs torch and transformers for every test in this directory.

`qa.py` reads torch while it is imported, to build `DTYPES`, and imports two
classes from transformers. The tests need neither for real: the model is
monkeypatched out and the boundary code is pure Python, so CI does not pull a
multi-hundred-MB wheel to test span arithmetic and the response envelope. This
follows test/unit/qwen36, which stubs vLLM the same way.

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

import sys
from unittest.mock import MagicMock

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
    for name, attributes in _REQUIRED.items():
        module = sys.modules.setdefault(name, _stub_package(name))
        for attribute in attributes:
            if not hasattr(module, attribute):
                setattr(module, attribute, MagicMock(name=f"{name}.{attribute}"))


def pytest_collectstart(collector):
    """Install the stubs just before pytest imports a module in this directory."""
    _install_stubs()


_install_stubs()

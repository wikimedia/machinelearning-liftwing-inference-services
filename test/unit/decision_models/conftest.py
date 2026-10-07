"""Stubs torch for every test in this directory.

`model.py` imports torch at module level and uses `torch.inference_mode()` in
`predict`. The tests need neither for real: the model is never loaded, so CI
does not pull a ROCm wheel to test payload validation and an image resize.
This follows test/unit/semantic_highlighting, which stubs torch and
transformers the same way, and test/unit/qwen36, which stubs vLLM.

Pillow is deliberately not stubbed. These tests decode real PNGs and check the
downscale and the decompression bomb guard, all of which a stub would make
vacuous.

`joint_schema_model` is not stubbed either. `model.py` imports it inside
`load()` rather than at module level, so nothing here touches it.

Two details are copied from test/unit/semantic_highlighting, and both come
from directories mocking the same module:

1. The stubs are installed from a hook, not only when this file is imported.
   pytest loads every conftest at startup, before importing any test module,
   so a stub installed at import time is in place too early to survive:
   test/unit/embeddings/test_model_transformers.py replaces
   `sys.modules["torch"]` when *its* module is imported, which is later.
   `pytest_collectstart` runs immediately before pytest imports each module
   here, which is after every other directory has had its say.

2. A module another directory installed is filled in, not replaced. This asks
   only for the names `model.py` reads and adds the ones that are absent, so
   whatever another directory set, it keeps.
"""

import contextlib
import sys
from unittest.mock import MagicMock

# What model.py reads from torch. `inference_mode` has to behave as a context
# manager rather than return a MagicMock, because `predict` enters it together
# with the inference lock: a stub that does not support the protocol would make
# every predict test fail for the wrong reason.
_REQUIRED = {"torch": {"inference_mode": contextlib.nullcontext}}


def _stub_package(name):
    """A stand-in module that answers any attribute, and `from x import y`."""
    module = MagicMock()
    module.__path__ = []
    module.__name__ = name
    return module


def _install_stubs():
    for name, attributes in _REQUIRED.items():
        module = sys.modules.setdefault(name, _stub_package(name))
        for attribute, value in attributes.items():
            if not hasattr(module, attribute) or isinstance(
                getattr(module, attribute), MagicMock
            ):
                setattr(module, attribute, value)


def pytest_collectstart(collector):
    """Install the stubs just before pytest imports a module in this directory."""
    _install_stubs()


_install_stubs()

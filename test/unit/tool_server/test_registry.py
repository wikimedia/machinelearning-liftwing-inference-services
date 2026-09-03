"""
Registry tests.

Auto-discovery trades explicitness for convenience, so these tests pin
the parts that would otherwise fail quietly: a tool that cannot be
imported must break startup, two tools cannot share a name, and the
published order must be deterministic.
"""

import sys
import textwrap

import pytest

from tool_server import registry


class TestToolSpec:
    def test_path_uses_hyphens_and_name_stays_the_function_name(self):
        def some_tool() -> dict:
            """Docstring."""

        spec = registry.ToolSpec(name="some_tool", fn=some_tool)
        assert spec.path == "/v1/tools/some-tool"
        assert spec.name == "some_tool"

    def test_description_is_the_docstring(self):
        def t() -> dict:
            """First line.

            More detail.
            """

        assert registry.ToolSpec(name="t", fn=t).description.startswith("First line.")


class TestRegistration:
    def test_decorator_registers_and_returns_the_function(self):
        registry_snapshot = dict(registry._REGISTRY)
        try:

            @registry.tool
            def demo_tool(x: int = 1) -> dict:
                """Demo tool."""
                return {"x": x}

            assert demo_tool(2) == {"x": 2}
            assert "demo_tool" in {t.name for t in registry.registered()}
        finally:
            registry._REGISTRY.clear()
            registry._REGISTRY.update(registry_snapshot)

    def test_duplicate_name_is_refused(self):
        registry_snapshot = dict(registry._REGISTRY)
        try:

            @registry.tool(name="clash")
            def first() -> dict:
                """First."""

            with pytest.raises(RuntimeError, match="Two tools are registered"):

                @registry.tool(name="clash")
                def second() -> dict:
                    """Second."""

        finally:
            registry._REGISTRY.clear()
            registry._REGISTRY.update(registry_snapshot)

    def test_missing_docstring_is_refused(self):
        with pytest.raises(RuntimeError, match="no docstring"):

            @registry.tool
            def undocumented() -> dict:
                return {}

    def test_re_registering_the_same_function_is_idempotent(self):
        """
        Discovery may import a module twice (package plus submodule);
        that must not look like a name clash.
        """
        registry_snapshot = dict(registry._REGISTRY)
        try:

            @registry.tool
            def stable_tool() -> dict:
                """Stable."""

            registry.tool(stable_tool)  # second registration, same object
            names = [t.name for t in registry.registered()]
            assert names.count("stable_tool") == 1
        finally:
            registry._REGISTRY.clear()
            registry._REGISTRY.update(registry_snapshot)


class TestDiscovery:
    def test_finds_the_toolbelt_in_deterministic_order(self):
        specs = registry.discover()
        assert [s.name for s in specs] == [
            "current_date",
            "wikipedia_semantic_search",
        ]

    def test_discovery_is_repeatable(self):
        assert [s.name for s in registry.discover()] == [
            s.name for s in registry.discover()
        ]

    def test_broken_tool_module_fails_loudly(self, tmp_path, monkeypatch):
        """
        A tool that cannot be imported must break discovery, not
        vanish from the toolbelt: pkgutil.walk_packages swallows import
        errors unless onerror re-raises.
        """
        pkg = tmp_path / "broken_toolbelt"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("")
        (pkg / "bad_tool.py").write_text(
            textwrap.dedent(
                """
                import module_that_does_not_exist_anywhere  # noqa: F401
                """
            )
        )
        monkeypatch.syspath_prepend(str(tmp_path))
        for mod in [m for m in sys.modules if m.startswith("broken_toolbelt")]:
            del sys.modules[mod]

        with pytest.raises(ModuleNotFoundError):
            registry.discover("broken_toolbelt")

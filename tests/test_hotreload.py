"""hotreload.py on throw-away modules: instances pick up new method bodies, from-imports in a
sibling module are rebound, and unrelated objects are left alone."""
from __future__ import annotations

import importlib
import sys
import tempfile
import unittest
from pathlib import Path

import _path  # noqa: F401

import hotreload


class TestHotReload(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name).resolve()
        sys.path.insert(0, str(self.root))
        (self.root / "hr_core.py").write_text(
            "class Thing:\n    def __init__(self):\n        self.n = 5\n    def value(self):\n        return self.n * 1\n\n"
            "def helper(x):\n    return x + 1\n", "utf-8")
        (self.root / "hr_user.py").write_text(
            "from hr_core import helper, Thing\n\nclass Holder:\n    def __init__(self):\n        self.thing = Thing()\n"
            "        self.things = {'a': Thing()}\n        self.list = [Thing()]\n    def use(self):\n        return helper(1)\n", "utf-8")
        for m in ("hr_core", "hr_user"):
            sys.modules.pop(m, None)
        self.core = importlib.import_module("hr_core")
        self.user = importlib.import_module("hr_user")

    def tearDown(self):
        sys.path.remove(str(self.root))
        for m in ("hr_core", "hr_user"):
            sys.modules.pop(m, None)
        self._tmp.cleanup()

    def test_project_modules_finds_only_root_files(self):
        mods = hotreload.project_modules(self.root)
        self.assertEqual(set(mods), {"hr_core", "hr_user"})

    def test_reload_rebinds_instances_and_imports(self):
        holder = self.user.Holder()
        self.assertEqual(holder.thing.value(), 5)
        self.assertEqual(holder.use(), 2)
        (self.root / "hr_core.py").write_text(
            "class Thing:\n    def __init__(self):\n        self.n = 5\n    def value(self):\n        return self.n * 10\n\n"
            "def helper(x):\n    return x + 100\n", "utf-8")
        info = hotreload.reload_module("hr_core", [holder], self.root, log=lambda *_: None)
        self.assertEqual(info["instances"], 3)                      # thing, things['a'], list[0]
        self.assertIn("hr_user.helper", info["imports"]); self.assertIn("hr_user.Thing", info["imports"])
        self.assertEqual(holder.thing.value(), 50)                  # new method, old attribute kept
        self.assertEqual(holder.things["a"].value(), 50)
        self.assertEqual(holder.list[0].value(), 50)
        self.assertEqual(holder.use(), 101)                         # from-import rebound
        self.assertEqual(self.user.Thing().value(), 50)             # new instances use the new class

    def test_unknown_module(self):
        with self.assertRaises(ValueError):
            hotreload.reload_module("hr_nope", [], self.root, log=lambda *_: None)

    def test_reachable_filters_by_module(self):
        holder = self.user.Holder()
        found = hotreload.reachable([holder], project_names={"hr_core"})
        self.assertEqual(len(found), 3)
        self.assertTrue(all(type(o).__name__ == "Thing" for o in found))


if __name__ == "__main__":
    unittest.main()

"""Swap a changed project module into the running process.

`importlib.reload` gives the module new code, but three things keep pointing at the old
code: live instances (their `__class__` is the old class object), `from x import y`
bindings in other modules, and threads already inside an old function. This module fixes
the first two:

  * every live object reachable from the roots whose class came from the reloaded module
    gets `obj.__class__ = <new class>` (methods change, instance attributes are kept);
  * every other project module that imported a name from it gets that name rebound.

A thread sitting in an old loop (`CytonBoard._run`, `ProtocolRunner._run`) keeps the old
body until it returns; anything it calls through `self.` uses the new code at once. `app.py`
cannot be reloaded (it is `__main__`); changes there need a restart.
"""
from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

SKIP_TYPES = (str, bytes, int, float, bool, type(None), types.FunctionType, types.ModuleType, type)


def project_modules(root: Path) -> dict[str, types.ModuleType]:
    """Loaded modules whose file sits directly in `root` (the project's flat layout)."""
    root = root.resolve()
    out = {}
    for name, mod in list(sys.modules.items()):
        f = getattr(mod, "__file__", None)
        if not f or name == "__main__":
            continue
        try:
            p = Path(f).resolve()
        except OSError:
            continue
        if p.parent == root and p.suffix == ".py":
            out[name] = mod
    return out


def reachable(roots, max_depth: int = 4, project_names: set[str] | None = None) -> list:
    """Objects reachable from `roots` through attributes, lists, tuples and dict values whose
    class was defined in one of `project_names` (all modules when None)."""
    seen: set[int] = set()
    found = []
    stack = [(r, 0) for r in roots if r is not None]
    while stack:
        obj, depth = stack.pop()
        if id(obj) in seen or isinstance(obj, SKIP_TYPES):
            continue
        seen.add(id(obj))
        mod = type(obj).__module__
        if project_names is None or mod in project_names:
            found.append(obj)
        if depth >= max_depth:
            continue
        children = []
        if isinstance(obj, dict):
            children = list(obj.values())
        elif isinstance(obj, (list, tuple, set, frozenset)):
            children = list(obj)
        else:
            d = getattr(obj, "__dict__", None)
            if isinstance(d, dict):
                children = list(d.values())
        for c in children:
            try:
                hash(id(c))
            except Exception:  # noqa: BLE001
                continue
            if type(c).__module__ in ("numpy", "threading", "queue", "collections") and not isinstance(c, (dict, list, tuple)):
                continue
            stack.append((c, depth + 1))
    return found


def reload_module(name: str, roots, root: Path, log=print) -> dict:
    """Reload `name`, rebind reachable instances and from-imports. Returns what was done."""
    mods = project_modules(root)
    if name not in sys.modules:
        raise ValueError(f"{name} is not loaded")
    mod = sys.modules[name]
    old_classes = {k: v for k, v in vars(mod).items() if isinstance(v, type) and v.__module__ == name}
    importlib.reload(mod)
    rebound_instances = 0
    for obj in reachable(roots, project_names={name}):
        cls = type(obj)
        new = getattr(mod, cls.__name__, None)
        if isinstance(new, type) and new is not cls:
            try:
                obj.__class__ = new
                rebound_instances += 1
            except TypeError as e:            # layout conflict (slots etc.)
                log(f"[hotreload] could not rebind {cls.__name__}: {e}")
    rebound_names = []
    for other_name, other in mods.items():
        if other_name == name:
            continue
        for attr, val in list(vars(other).items()):
            if getattr(val, "__module__", None) != name:
                continue
            if not isinstance(val, (type, types.FunctionType)):
                continue
            new = getattr(mod, getattr(val, "__name__", attr), None)
            if new is not None and new is not val:
                setattr(other, attr, new)
                rebound_names.append(f"{other_name}.{attr}")
    log(f"[hotreload] {name}: {rebound_instances} instances, {len(rebound_names)} imports rebound")
    return {"module": name, "instances": rebound_instances, "imports": rebound_names,
            "classes": sorted(old_classes)}

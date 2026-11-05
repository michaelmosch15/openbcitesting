"""Astra in-flight fixer: the operator reports an issue, Astra reads the source, patches it,
the patch is verified (compile, node --check, the unit tests), backed up, and swapped into the
running server (hotreload.py) or flagged as needing a restart.

One job at a time. The conversation with the model is a loop of JSON actions:

    {"action": "read",  "files": ["pipeline.py", "static/app.js"], "why": "..."}
    {"action": "patch", "summary": "...", "needs_restart": false,
     "edits": [{"file": "pipeline.py", "find": "<exact, unique text>", "replace": "<new text>"}]}
    {"action": "answer", "text": "..."}        # no code change warranted / needs the operator

Guard rails: only files under the project (root *.py, static/, tests/, docs/*.md, config.json,
protocols.json); `.env`, sessions/ and anything outside are refused. Without `allow_prereg`
the pre-registered parameters (protocols.json, index weights, artifact thresholds, gates,
block durations) cannot be touched. A patch whose verification fails is rolled back and the
failure is handed back to the model for one more try. Every applied fix is recorded under
astra_fixes/<stamp>_<slug>/ with before/ and after/ copies, the diff and fix.md; `revert()`
restores the last fix from its before/ copies.
"""
from __future__ import annotations

import difflib
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
FIXES_DIR = HERE / "astra_fixes"
MAX_ROUNDS = 8
MAX_PATCH_TRIES = 2
MAX_FILE_CHARS = 60000
MAX_EDITS = 12
ALLOWED_SUFFIX = {".py", ".js", ".html", ".css", ".json", ".md", ".ps1", ".bat", ".txt"}
DENY_FILES = {".env", "requirements.txt"}
PREREG_FILES = {"protocols.json"}
PREREG_MARKERS = ("BLOCK_MIN_N", "BLOCK_MIN_RHO", "BLOCK_MIN_S", "ACCURACY_FLOW_RANGE", "DEFAULT_WEIGHTS",
                  "\"theta\":", "\"alpha\":", "\"beta\":", "artifact_p2p_uv", "artifact_abs_uv", "index_window_s",
                  "index_seg_s", "index_min_clean_fraction", "warmup_epochs", "baseline_epochs", "duration_s",
                  "INDEX_BANDS", "TOTAL_BAND")
NO_RELOAD = {"app"}                 # __main__; a restart is the only way
RESTART_FILES = {"config.json"}     # read once at start-up

FIXER_SYSTEM = """You are Astra, the in-flight engineer of a running localhost app: the OpenBCI x BioProxal flow-validation dashboard (Python 3.12+ stdlib + numpy server in the project root, vanilla JS/Canvas browser in static/, unit tests in tests/, contract in docs/API.md). The operator reports an issue while the app runs. You read what you need, then patch it with the smallest correct change; the server verifies (py_compile, node --check, python -m unittest discover -s tests) and hot-swaps the modules.

Reply with exactly ONE JSON object per turn and nothing else:
  {"action":"read","files":["<path>", ...],"why":"<one line>"}            at most 6 files per turn
  {"action":"patch","summary":"<what and why, 1-3 sentences>","needs_restart":false,
   "edits":[{"file":"<path>","find":"<exact text copied from the file, occurring exactly once>","replace":"<replacement>"}, ...]}
  {"action":"answer","text":"<when no code change is right: explain, or ask the operator one precise question>"}

Rules: read before you patch (you only know a file after reading it this job). `find` must be copied verbatim (whitespace included) and match once; keep edits minimal and in the project's style; update docs/API.md and tests when behaviour changes; never touch .env, sessions/ or files outside the project; no new dependencies. Pre-registered parameters (index weights, artifact thresholds, verdict gates, protocol durations) are frozen unless the operator explicitly allowed it in this job. If the verification fails you get the output back: fix it or answer with what blocks you. Do not praise or pad."""


def _rel(path: str) -> str:
    return re.sub(r"^(\./)+", "", str(path).replace("\\", "/").strip())


def resolve_path(rel: str) -> Path:
    """A project-relative path that is allowed to be read or written; ValueError otherwise."""
    rel = _rel(rel)
    if not rel or rel.startswith("/") or ".." in rel.split("/") or re.match(r"^[A-Za-z]:", rel):
        raise ValueError(f"refused path {rel!r}")
    p = (HERE / rel).resolve()
    if HERE not in p.parents and p != HERE:
        raise ValueError(f"refused path {rel!r}")
    parts = p.relative_to(HERE).parts
    if parts[0] in ("sessions", "astra_fixes", ".git", ".claude", "__pycache__") or p.name in DENY_FILES or p.suffix == ".env":
        raise ValueError(f"refused path {rel!r}")
    if p.suffix not in ALLOWED_SUFFIX:
        raise ValueError(f"refused file type {rel!r}")
    if len(parts) > 1 and parts[0] not in ("static", "tests", "docs"):
        raise ValueError(f"refused folder {rel!r}")
    return p


def file_tree() -> list[dict]:
    out = []
    for p in sorted(HERE.glob("*.py")) + sorted((HERE / "static").glob("*")) + sorted((HERE / "tests").glob("*.py")) \
            + sorted((HERE / "docs").glob("*.md")) + [HERE / "config.json", HERE / "protocols.json"]:
        if p.is_file():
            try:
                with open(p, "rb") as f:
                    n = sum(1 for _ in f)
            except OSError:
                n = 0
            out.append({"path": _rel(str(p.relative_to(HERE))), "lines": n})
    return out


def read_file(rel: str) -> str:
    p = resolve_path(rel)
    if not p.exists():
        return f"(missing: {rel})"
    txt = p.read_text("utf-8", errors="replace")
    if len(txt) > MAX_FILE_CHARS:
        half = MAX_FILE_CHARS // 2
        txt = txt[:half] + f"\n\n... [{len(txt) - MAX_FILE_CHARS} chars omitted; ask for the file again with a narrower need] ...\n\n" + txt[-half:]
    return txt


def parse_action(text: str) -> dict:
    s = text.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z]*\n?", "", s).rstrip("`").strip()
    i, j = s.find("{"), s.rfind("}")
    if i < 0 or j <= i:
        raise ValueError("no JSON object in the reply")
    d = json.loads(s[i:j + 1])
    if not isinstance(d, dict) or d.get("action") not in ("read", "patch", "answer"):
        raise ValueError("reply must be a JSON object with action read|patch|answer")
    return d


def apply_edits(edits: list[dict], allow_prereg: bool) -> dict:
    """Validate and apply find/replace edits. Returns {files: {rel: (before, after)}}; raises
    ValueError before writing anything when any edit is bad."""
    if not isinstance(edits, list) or not edits:
        raise ValueError("patch has no edits")
    if len(edits) > MAX_EDITS:
        raise ValueError(f"more than {MAX_EDITS} edits")
    texts: dict[str, str] = {}
    before: dict[str, str] = {}
    for e in edits:
        rel = _rel(str(e.get("file") or ""))
        p = resolve_path(rel)
        if rel in PREREG_FILES and not allow_prereg:
            raise ValueError(f"{rel} holds pre-registered parameters; the operator did not allow changing them")
        if rel not in texts:
            texts[rel] = p.read_text("utf-8") if p.exists() else ""
            before[rel] = texts[rel]
        find, repl = e.get("find"), e.get("replace")
        if not isinstance(find, str) or not isinstance(repl, str) or not find:
            raise ValueError(f"edit for {rel}: find/replace must be non-empty strings")
        n = texts[rel].count(find)
        if n != 1:
            raise ValueError(f"edit for {rel}: `find` occurs {n} times, must be exactly once")
        if not allow_prereg:
            changed = [ln for ln in difflib.ndiff(find.splitlines(), repl.splitlines()) if ln[:1] in "+-"]
            hit = [m for m in PREREG_MARKERS if any(m in ln for ln in changed)]
            if hit:
                raise ValueError(f"edit for {rel} changes pre-registered parameter(s) {hit}; the operator did not allow that")
        texts[rel] = texts[rel].replace(find, repl, 1)
    for rel, txt in texts.items():
        resolve_path(rel).write_text(txt, "utf-8", newline="\n")
    return {"files": {rel: (before[rel], texts[rel]) for rel in texts}}


def unified_diff(files: dict[str, tuple[str, str]]) -> str:
    out = []
    for rel, (a, b) in files.items():
        out.extend(difflib.unified_diff(a.splitlines(), b.splitlines(), f"a/{rel}", f"b/{rel}", lineterm="", n=3))
    return "\n".join(out)


def verify(changed: list[str], run_tests: bool = True, timeout: float = 240.0) -> dict:
    """Compile changed .py, node --check changed .js, run the unit tests."""
    steps = []
    ok = True
    for rel in changed:
        p = HERE / rel
        if p.suffix == ".py":
            r = subprocess.run([sys.executable, "-m", "py_compile", str(p)], capture_output=True, text=True, cwd=HERE, timeout=60)
            steps.append({"step": f"py_compile {rel}", "ok": r.returncode == 0, "out": (r.stdout + r.stderr)[-2000:]})
            ok &= r.returncode == 0
        elif p.suffix == ".js":
            node = shutil.which("node")
            if node:
                r = subprocess.run([node, "--check", str(p)], capture_output=True, text=True, cwd=HERE, timeout=60)
                steps.append({"step": f"node --check {rel}", "ok": r.returncode == 0, "out": (r.stdout + r.stderr)[-2000:]})
                ok &= r.returncode == 0
            else:
                steps.append({"step": f"node --check {rel}", "ok": True, "out": "node not installed; skipped"})
        elif p.suffix == ".json":
            try:
                json.loads(p.read_text("utf-8"))
                steps.append({"step": f"json {rel}", "ok": True, "out": ""})
            except ValueError as e:
                steps.append({"step": f"json {rel}", "ok": False, "out": str(e)})
                ok = False
    if ok and run_tests and any(not rel.startswith("docs/") for rel in changed):
        try:
            r = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests"], capture_output=True, text=True,
                               cwd=HERE, timeout=timeout)
            out = (r.stdout + r.stderr)
            steps.append({"step": "unittest", "ok": r.returncode == 0, "out": out[-4000:]})
            ok &= r.returncode == 0
        except subprocess.TimeoutExpired:
            steps.append({"step": "unittest", "ok": False, "out": f"timed out after {timeout:.0f} s"})
            ok = False
    return {"ok": ok, "steps": steps}


class Fixer:
    def __init__(self, cfg: dict, emit, chat_fn, context_fn, reload_fn, log=print, verify_fn=verify,
                 restart_fn=None, protocol_reload_fn=None):
        """context_fn() -> str (live snapshot, setup state, log tail, notes); reload_fn(module_name) -> dict;
        restart_fn() restarts the server; protocol_reload_fn() re-reads protocols.json."""
        self.cfg, self.emit, self.chat_fn, self.context_fn, self.reload_fn, self.log = cfg, emit, chat_fn, context_fn, reload_fn, log
        self.verify_fn, self.restart_fn, self.protocol_reload_fn = verify_fn, restart_fn, protocol_reload_fn
        self.model = str(cfg.get("model", "gpt-6-astra"))
        self.max_tokens = int(cfg.get("fix_max_tokens", 16000))
        self.busy = False
        self.error: str | None = None
        self.fixes: list[dict] = []           # applied fixes, newest last
        self.needs_restart = False
        self.reload_browser = False
        self._lock = threading.Lock()
        self._id = 0

    # --- helpers ---
    def _emit(self, job: str, phase: str, text: str = "", **kw) -> None:
        self.emit({"kind": "fix", "id": job, "t": time.time(), "phase": phase, "text": text, **kw})

    def _ask(self, msgs: list[dict]) -> dict:
        reply = self.chat_fn(msgs, model=self.model, max_tokens=self.max_tokens)
        msgs.append({"role": "assistant", "content": reply})
        return parse_action(reply)

    def _record(self, job: str, issue: str, summary: str, files: dict, diff: str, ver: dict, reload_info: list) -> Path:
        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        slug = re.sub(r"[^A-Za-z0-9]+", "-", issue)[:40].strip("-").lower() or "fix"
        d = FIXES_DIR / f"{stamp}_{slug}"
        for rel, (a, b) in files.items():
            (d / "before" / rel).parent.mkdir(parents=True, exist_ok=True)
            (d / "after" / rel).parent.mkdir(parents=True, exist_ok=True)
            (d / "before" / rel).write_text(a, "utf-8", newline="\n")
            (d / "after" / rel).write_text(b, "utf-8", newline="\n")
        steps = "\n".join(f"- {s['step']}: {'ok' if s['ok'] else 'FAILED'}" for s in ver["steps"])
        reloads = "\n".join(f"- {r}" for r in reload_info) or "- none"
        (d / "fix.md").write_text(
            f"# Astra fix {stamp} ({self.model})\n\n**Issue:** {issue}\n\n**Summary:** {summary}\n\n"
            f"**Files:** {', '.join(files)}\n\n## Verification\n{steps}\n\n## Reload\n{reloads}\n\n## Diff\n```diff\n{diff}\n```\n", "utf-8")
        (d / "fix.json").write_text(json.dumps({"id": job, "t": time.time(), "issue": issue, "summary": summary,
                                                "files": list(files), "verify": ver, "reload": reload_info}, indent=1), "utf-8")
        return d

    def _swap_in(self, files: dict) -> tuple[list[str], bool, bool]:
        """Hot-reload what can be reloaded; report restart / browser-reload needs."""
        info, restart, browser = [], False, False
        for rel in files:
            p = Path(rel)
            if rel in RESTART_FILES:
                restart = True
                info.append(f"{rel}: read at start-up, restart needed")
            elif rel == "protocols.json":
                if self.protocol_reload_fn:
                    try:
                        self.protocol_reload_fn()
                        info.append("protocols.json re-read")
                    except Exception as e:  # noqa: BLE001
                        info.append(f"protocols.json: {e}")
            elif p.parts[0] == "static":
                browser = True
                info.append(f"{rel}: served fresh on the next page load")
            elif p.parts[0] in ("tests", "docs"):
                info.append(f"{rel}: no reload needed")
            elif p.suffix == ".py":
                mod = p.stem
                if mod in NO_RELOAD:
                    restart = True
                    info.append(f"{rel}: is __main__, restart needed")
                else:
                    try:
                        r = self.reload_fn(mod)
                        info.append(f"{rel}: reloaded ({r.get('instances', 0)} instances, {len(r.get('imports', []))} imports rebound)")
                    except Exception as e:  # noqa: BLE001
                        restart = True
                        info.append(f"{rel}: reload failed ({e}), restart needed")
        return info, restart, browser

    # --- the job ---
    def fix(self, issue: str, allow_prereg: bool = False) -> dict:
        issue = str(issue).strip()[:4000]
        if not issue:
            raise ValueError("describe the issue")
        with self._lock:
            self.busy = True
            self._id += 1
            job = f"f{int(time.time())}-{self._id}"
            try:
                return self._run(job, issue, bool(allow_prereg))
            except Exception as e:  # noqa: BLE001
                self.error = str(e)
                self._emit(job, "failed", str(e), done=True)
                raise
            finally:
                self.busy = False

    def _run(self, job: str, issue: str, allow_prereg: bool) -> dict:
        self.error = None
        self._emit(job, "start", issue, issue=issue)
        tree = "\n".join(f"{f['path']} ({f['lines']} lines)" for f in file_tree())
        named = sorted({m for m in re.findall(r"\b[\w./-]+\.(?:py|js|html|css|json|md)\b", issue)})
        pre = []
        read_so_far: set[str] = set()
        for rel in named[:4]:
            try:
                pre.append(f"===== FILE {_rel(rel)} =====\n{read_file(rel)}")
                read_so_far.add(_rel(rel))
            except ValueError:
                pass
        ctx = self.context_fn() if self.context_fn else ""
        user = (f"ISSUE REPORTED BY THE OPERATOR:\n{issue}\n\n"
                f"Pre-registered parameters may be changed in this job: {'YES' if allow_prereg else 'NO'}\n\n"
                f"PROJECT FILES:\n{tree}\n\nLIVE CONTEXT:\n{ctx}\n\n" + ("\n\n".join(pre) + "\n\n" if pre else "") + "Your first action.")
        msgs = [{"role": "system", "content": FIXER_SYSTEM}, {"role": "user", "content": user}]
        patch_tries = 0
        for rnd in range(MAX_ROUNDS):
            try:
                act = self._ask(msgs)
            except ValueError as e:
                msgs.append({"role": "user", "content": f"Your reply could not be used: {e}. Reply with one JSON object only."})
                continue
            if act["action"] == "read":
                files = [str(f) for f in (act.get("files") or [])][:6]
                self._emit(job, "read", "reading " + ", ".join(files), files=files)
                parts = []
                for rel in files:
                    try:
                        parts.append(f"===== FILE {_rel(rel)} =====\n{read_file(rel)}")
                        read_so_far.add(_rel(rel))
                    except ValueError as e:
                        parts.append(f"===== FILE {rel} =====\n(refused: {e})")
                msgs.append({"role": "user", "content": "\n\n".join(parts) + "\n\nNext action."})
                continue
            if act["action"] == "answer":
                text = str(act.get("text") or "").strip()
                self._emit(job, "answer", text, done=True)
                return {"id": job, "status": "answer", "text": text}
            # patch
            patch_tries += 1
            summary = str(act.get("summary") or "").strip()
            edits = act.get("edits") or []
            unread = sorted({_rel(str(e.get("file") or "")) for e in edits} - read_so_far)
            self._emit(job, "patch", f"applying {len(edits)} edit(s): {summary}", files=sorted({_rel(str(e.get('file') or '')) for e in edits}))
            try:
                if unread:
                    raise ValueError(f"you patched {unread} without reading them this job; read first")
                applied = apply_edits(edits, allow_prereg)
            except ValueError as e:
                self._emit(job, "rejected", str(e))
                if patch_tries >= MAX_PATCH_TRIES:
                    self._emit(job, "failed", f"patch rejected: {e}", done=True)
                    return {"id": job, "status": "failed", "text": str(e)}
                msgs.append({"role": "user", "content": f"Patch rejected: {e}. Fix the patch (or answer)."})
                continue
            files = applied["files"]
            diff = unified_diff(files)
            self._emit(job, "verify", "compile, node --check, unit tests…", files=list(files))
            ver = self.verify_fn(list(files))
            if not ver["ok"]:
                for rel, (a, _) in files.items():
                    resolve_path(rel).write_text(a, "utf-8", newline="\n")
                failed = [s for s in ver["steps"] if not s["ok"]]
                msg = "\n".join(f"{s['step']} FAILED:\n{s['out']}" for s in failed)
                self._emit(job, "rejected", "verification failed, rolled back:\n" + msg[-1500:])
                if patch_tries >= MAX_PATCH_TRIES:
                    self._emit(job, "failed", "verification failed twice; files restored", done=True, diff=diff)
                    return {"id": job, "status": "failed", "text": msg[-2000:]}
                msgs.append({"role": "user", "content": f"The patch was applied and verification FAILED; the files were restored to their previous content:\n{msg[-6000:]}\n\nSend a corrected patch (re-read files if needed) or answer."})
                continue
            info, restart, browser = self._swap_in(files)
            restart = restart or bool(act.get("needs_restart"))
            self.needs_restart = self.needs_restart or restart
            self.reload_browser = self.reload_browser or browser
            rec_dir = self._record(job, issue, summary, files, diff, ver, info)
            rec = {"id": job, "t": time.time(), "issue": issue, "summary": summary, "files": list(files),
                   "record": str(rec_dir.relative_to(HERE)), "needs_restart": restart, "reload_browser": browser,
                   "status": "applied"}
            self.fixes.append(rec)
            tests = next((s for s in ver["steps"] if s["step"] == "unittest"), None)
            tests_line = ""
            if tests:
                m = re.search(r"Ran (\d+) tests?", tests["out"])
                tests_line = f"; tests: {m.group(0) if m else 'ok'}"
            self._emit(job, "done", summary + tests_line, done=True, summary=summary, diff=diff, files=list(files),
                       needs_restart=restart, reload_browser=browser, reload=info, record=rec["record"], verify=ver["steps"])
            self.log(f"[astra-fix] applied: {summary} -> {', '.join(files)}" + (" (RESTART NEEDED)" if restart else ""))
            return {**rec, "diff": diff}
        self._emit(job, "failed", f"no result after {MAX_ROUNDS} rounds", done=True)
        return {"id": job, "status": "failed", "text": "no result"}

    # --- undo / restart ---
    def revert(self) -> dict:
        with self._lock:
            if not self.fixes:
                raise ValueError("nothing to revert")
            rec = self.fixes.pop()
            d = HERE / rec["record"]
            files = {}
            for rel in rec["files"]:
                src = d / "before" / rel
                p = resolve_path(rel)
                cur = p.read_text("utf-8") if p.exists() else ""
                txt = src.read_text("utf-8")
                p.write_text(txt, "utf-8", newline="\n")
                files[rel] = (cur, txt)
            info, restart, browser = self._swap_in(files)
            self.needs_restart = self.needs_restart or restart
            self.reload_browser = self.reload_browser or browser
            job = rec["id"] + "r"
            self._emit(job, "done", f"reverted: {rec['summary']}", done=True, summary="revert: " + rec["summary"],
                       diff=unified_diff(files), files=list(files), needs_restart=restart, reload_browser=browser, reload=info)
            self.log(f"[astra-fix] reverted {rec['record']}")
            return {"reverted": rec["record"], "files": list(files), "needs_restart": restart}

    def restart(self) -> dict:
        if self.restart_fn is None:
            raise ValueError("restart is not wired")
        self.log("[astra-fix] restarting the server")
        threading.Timer(0.5, self.restart_fn).start()
        return {"restarting": True}

    def status(self) -> dict:
        return {"busy": self.busy, "error": self.error, "n_fixes": len(self.fixes), "needs_restart": self.needs_restart,
                "reload_browser": self.reload_browser, "model": self.model,
                "last": self.fixes[-1]["summary"] if self.fixes else None}

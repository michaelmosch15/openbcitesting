"""fixer.py against a sandbox project folder (fixer.HERE is pointed at a temp dir): path
rules, the JSON action parser, find/replace application with the pre-registration guard,
the diff, and the Fixer loop with a scripted fake model, fake verifier and fake reloader."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import _path  # noqa: F401

import fixer


class Sandbox(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name).resolve()
        (root / "static").mkdir(); (root / "tests").mkdir(); (root / "docs").mkdir(); (root / "sessions").mkdir()
        (root / "pipeline.py").write_text("def f(x):\n    return x + 1\n\nVALUE = 2\n", "utf-8")
        (root / "static" / "app.js").write_text("const a = 1;\n", "utf-8")
        (root / "docs" / "API.md").write_text("# api\n", "utf-8")
        (root / "config.json").write_text('{"flow_index": {"theta": 1.0, "alpha": 0.0}}\n', "utf-8")
        (root / "protocols.json").write_text('{"p": {"blocks": [{"duration_s": 60}]}}\n', "utf-8")
        (root / "report.py").write_text("BLOCK_MIN_RHO = 0.5\nX = 1\n", "utf-8")
        (root / ".env").write_text("OPENAI_KEY='x'\n", "utf-8")
        (root / "sessions" / "s.csv").write_text("a\n", "utf-8")
        self.root = root
        self._old = (fixer.HERE, fixer.FIXES_DIR)
        fixer.HERE, fixer.FIXES_DIR = root, root / "astra_fixes"

    def tearDown(self):
        fixer.HERE, fixer.FIXES_DIR = self._old
        self._tmp.cleanup()


class TestPaths(Sandbox):
    def test_allowed_and_refused(self):
        self.assertEqual(fixer.resolve_path("pipeline.py"), self.root / "pipeline.py")
        self.assertEqual(fixer.resolve_path("static/app.js"), self.root / "static" / "app.js")
        self.assertEqual(fixer.resolve_path("./docs/API.md"), self.root / "docs" / "API.md")
        for bad in (".env", "sessions/s.csv", "../x.py", "C:/x.py", "/etc/passwd", "astra_fixes/a.md", "other/x.py",
                    "pipeline.exe", "requirements.txt", ""):
            with self.assertRaises(ValueError, msg=bad):
                fixer.resolve_path(bad)

    def test_file_tree_and_read(self):
        tree = {f["path"]: f["lines"] for f in fixer.file_tree()}
        self.assertEqual(tree["pipeline.py"], 4)
        self.assertIn("static/app.js", tree)
        self.assertNotIn(".env", tree)
        self.assertIn("return x + 1", fixer.read_file("pipeline.py"))
        self.assertTrue(fixer.read_file("nope.py").startswith("(missing"))
        big = self.root / "big.py"
        big.write_text("x = 1\n" * 20000, "utf-8")
        self.assertIn("chars omitted", fixer.read_file("big.py"))


class TestParse(unittest.TestCase):
    def test_parse_action(self):
        self.assertEqual(fixer.parse_action('{"action":"read","files":["a.py"]}')["files"], ["a.py"])
        self.assertEqual(fixer.parse_action('```json\n{"action":"answer","text":"hi"}\n```')["text"], "hi")
        self.assertEqual(fixer.parse_action('Sure.\n{"action":"patch","edits":[]}\n')["action"], "patch")
        for bad in ("nothing", '{"action":"delete"}', '{"x": 1}', "[1,2]"):
            with self.assertRaises(ValueError):
                fixer.parse_action(bad)


class TestApplyEdits(Sandbox):
    def test_apply_unique_match(self):
        r = fixer.apply_edits([{"file": "pipeline.py", "find": "return x + 1", "replace": "return x + 2"}], False)
        self.assertEqual((self.root / "pipeline.py").read_text("utf-8"), "def f(x):\n    return x + 2\n\nVALUE = 2\n")
        before, after = r["files"]["pipeline.py"]
        self.assertIn("x + 1", before); self.assertIn("x + 2", after)
        diff = fixer.unified_diff(r["files"])
        self.assertIn("-    return x + 1", diff); self.assertIn("+    return x + 2", diff)

    def test_nothing_written_when_one_edit_is_bad(self):
        with self.assertRaises(ValueError):
            fixer.apply_edits([{"file": "pipeline.py", "find": "return x + 1", "replace": "return x + 2"},
                               {"file": "pipeline.py", "find": "not there", "replace": "y"}], False)
        self.assertIn("x + 1", (self.root / "pipeline.py").read_text("utf-8"))

    def test_ambiguous_and_empty(self):
        (self.root / "pipeline.py").write_text("a = 1\na = 1\n", "utf-8")
        with self.assertRaises(ValueError):
            fixer.apply_edits([{"file": "pipeline.py", "find": "a = 1", "replace": "a = 2"}], False)
        with self.assertRaises(ValueError):
            fixer.apply_edits([{"file": "pipeline.py", "find": "", "replace": "a"}], False)
        with self.assertRaises(ValueError):
            fixer.apply_edits([], False)
        with self.assertRaises(ValueError):
            fixer.apply_edits([{"file": "pipeline.py", "find": "a", "replace": "b"}] * 13, False)

    def test_prereg_guard(self):
        with self.assertRaises(ValueError):
            fixer.apply_edits([{"file": "protocols.json", "find": "60", "replace": "30"}], False)
        with self.assertRaises(ValueError):
            fixer.apply_edits([{"file": "report.py", "find": "BLOCK_MIN_RHO = 0.5", "replace": "BLOCK_MIN_RHO = 0.3"}], False)
        with self.assertRaises(ValueError):
            fixer.apply_edits([{"file": "config.json", "find": '"theta": 1.0', "replace": '"theta": 0.5'}], False)
        # a bug fix in the same file that does not touch a marker line is fine
        fixer.apply_edits([{"file": "report.py", "find": "X = 1", "replace": "X = 2"}], False)
        # and everything is allowed with the flag
        fixer.apply_edits([{"file": "protocols.json", "find": "60", "replace": "30"}], True)
        fixer.apply_edits([{"file": "report.py", "find": "BLOCK_MIN_RHO = 0.5", "replace": "BLOCK_MIN_RHO = 0.3"}], True)
        self.assertIn("0.3", (self.root / "report.py").read_text("utf-8"))

    def test_refused_targets(self):
        with self.assertRaises(ValueError):
            fixer.apply_edits([{"file": ".env", "find": "x", "replace": "y"}], True)
        with self.assertRaises(ValueError):
            fixer.apply_edits([{"file": "sessions/s.csv", "find": "a", "replace": "b"}], True)


class ScriptedChat:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def __call__(self, msgs, model, max_tokens):
        self.calls.append([dict(m) for m in msgs])
        return self.replies.pop(0) if self.replies else '{"action":"answer","text":"out of script"}'


def ok_verify(changed, run_tests=True):
    return {"ok": True, "steps": [{"step": "unittest", "ok": True, "out": "Ran 3 tests\nOK"}]}


class TestFixerLoop(Sandbox):
    def make(self, replies, verify=ok_verify, reload_fn=None):
        self.events = []
        self.reloaded = []
        chat = ScriptedChat(replies)
        f = fixer.Fixer({"model": "m"}, self.events.append, chat,
                        lambda: "LIVE CONTEXT HERE", reload_fn or (lambda name: self.reloaded.append(name) or {"instances": 1, "imports": []}),
                        log=lambda *_: None, verify_fn=verify, restart_fn=lambda: self.reloaded.append("RESTART"),
                        protocol_reload_fn=lambda: self.reloaded.append("protocols"))
        return f, chat

    def test_read_patch_done(self):
        f, chat = self.make([
            '{"action":"read","files":["pipeline.py","static/app.js",".env"],"why":"see it"}',
            json.dumps({"action": "patch", "summary": "off by one", "edits": [
                {"file": "pipeline.py", "find": "return x + 1", "replace": "return x + 2"},
                {"file": "static/app.js", "find": "const a = 1;", "replace": "const a = 2;"}]}),
        ])
        r = f.fix("f(1) returns 2 but should return 3; see pipeline.py")
        self.assertEqual(r["status"], "applied")
        self.assertEqual(sorted(r["files"]), ["pipeline.py", "static/app.js"])
        self.assertIn("x + 2", (self.root / "pipeline.py").read_text("utf-8"))
        # context + named file went into the first user message; refused .env reported, not read
        first = chat.calls[0][1]["content"]
        self.assertIn("LIVE CONTEXT HERE", first); self.assertIn("===== FILE pipeline.py", first)
        second = chat.calls[1][-1]["content"]
        self.assertIn("refused", second); self.assertIn("const a = 1;", second)
        # swap-in: module reloaded, static flagged for a browser reload
        self.assertEqual(self.reloaded, ["pipeline"])
        self.assertTrue(r["reload_browser"]); self.assertFalse(r["needs_restart"])
        phases = [e["phase"] for e in self.events]
        self.assertEqual(phases, ["start", "read", "patch", "verify", "done"])
        done = self.events[-1]
        self.assertIn("tests: Ran 3 tests", done["text"]); self.assertIn("-    return x + 1", done["diff"])
        # record on disk
        rec = self.root / r["record"]
        self.assertTrue((rec / "before" / "pipeline.py").exists())
        self.assertIn("x + 1", (rec / "before" / "pipeline.py").read_text("utf-8"))
        self.assertIn("x + 2", (rec / "after" / "pipeline.py").read_text("utf-8"))
        md = (rec / "fix.md").read_text("utf-8")
        self.assertIn("off by one", md); self.assertIn("```diff", md)
        self.assertEqual(f.status()["n_fixes"], 1)
        # revert restores and re-swaps
        rv = f.revert()
        self.assertEqual(sorted(rv["files"]), ["pipeline.py", "static/app.js"])
        self.assertIn("x + 1", (self.root / "pipeline.py").read_text("utf-8"))
        self.assertEqual(self.reloaded, ["pipeline", "pipeline"])
        self.assertEqual(f.status()["n_fixes"], 0)
        with self.assertRaises(ValueError):
            f.revert()

    def test_patch_without_read_is_rejected_then_fixed(self):
        f, chat = self.make([
            json.dumps({"action": "patch", "summary": "s", "edits": [{"file": "pipeline.py", "find": "VALUE = 2", "replace": "VALUE = 3"}]}),
            '{"action":"read","files":["pipeline.py"]}',
            json.dumps({"action": "patch", "summary": "s", "edits": [{"file": "pipeline.py", "find": "VALUE = 2", "replace": "VALUE = 3"}]}),
        ])
        r = f.fix("bump")
        self.assertEqual(r["status"], "applied")
        self.assertIn("without reading", chat.calls[1][-1]["content"])
        self.assertIn("rejected", [e["phase"] for e in self.events])

    def test_failed_verification_rolls_back_and_retries(self):
        calls = {"n": 0}

        def flaky(changed, run_tests=True):
            calls["n"] += 1
            if calls["n"] == 1:
                return {"ok": False, "steps": [{"step": "unittest", "ok": False, "out": "FAIL: test_x\nAssertionError"}]}
            return ok_verify(changed)
        f, chat = self.make([
            '{"action":"read","files":["pipeline.py"]}',
            json.dumps({"action": "patch", "summary": "try 1", "edits": [{"file": "pipeline.py", "find": "VALUE = 2", "replace": "VALUE = 99"}]}),
            json.dumps({"action": "patch", "summary": "try 2", "edits": [{"file": "pipeline.py", "find": "VALUE = 2", "replace": "VALUE = 3"}]}),
        ], verify=flaky)
        r = f.fix("value wrong")
        self.assertEqual(r["status"], "applied"); self.assertEqual(r["summary"], "try 2")
        self.assertIn("VALUE = 3", (self.root / "pipeline.py").read_text("utf-8"))
        self.assertIn("verification FAILED", chat.calls[2][-1]["content"])
        self.assertIn("AssertionError", chat.calls[2][-1]["content"])
        self.assertEqual([e["phase"] for e in self.events].count("rejected"), 1)

    def test_two_failures_give_up_with_files_restored(self):
        def bad(changed, run_tests=True):
            return {"ok": False, "steps": [{"step": "py_compile pipeline.py", "ok": False, "out": "SyntaxError"}]}
        f, _ = self.make([
            '{"action":"read","files":["pipeline.py"]}',
            json.dumps({"action": "patch", "summary": "a", "edits": [{"file": "pipeline.py", "find": "VALUE = 2", "replace": "VALUE = ("}]}),
            json.dumps({"action": "patch", "summary": "b", "edits": [{"file": "pipeline.py", "find": "VALUE = 2", "replace": "VALUE = )"}]}),
        ], verify=bad)
        r = f.fix("x")
        self.assertEqual(r["status"], "failed")
        self.assertEqual((self.root / "pipeline.py").read_text("utf-8"), "def f(x):\n    return x + 1\n\nVALUE = 2\n")
        self.assertEqual(self.events[-1]["phase"], "failed"); self.assertTrue(self.events[-1]["done"])
        self.assertEqual(f.status()["n_fixes"], 0)

    def test_answer_and_bad_json(self):
        f, chat = self.make(["I think the issue is elsewhere", '{"action":"answer","text":"Which panel shows it?"}'])
        r = f.fix("something odd")
        self.assertEqual(r["status"], "answer"); self.assertEqual(r["text"], "Which panel shows it?")
        self.assertIn("could not be used", chat.calls[1][-1]["content"])
        self.assertEqual(self.events[-1]["phase"], "answer")

    def test_prereg_rejected_without_flag_allowed_with(self):
        script = ['{"action":"read","files":["protocols.json"]}',
                  json.dumps({"action": "patch", "summary": "shorter", "edits": [{"file": "protocols.json", "find": "60", "replace": "30"}]}),
                  '{"action":"answer","text":"blocked"}']
        f, _ = self.make(list(script))
        r = f.fix("shorten blocks")
        self.assertEqual(r["status"], "answer")
        self.assertIn("rejected", [e["phase"] for e in self.events])
        f2, _ = self.make(list(script))
        r2 = f2.fix("shorten blocks", allow_prereg=True)
        self.assertEqual(r2["status"], "applied")
        self.assertIn("protocols", self.reloaded)

    def test_restart_flags(self):
        f, _ = self.make(['{"action":"read","files":["config.json"]}',
                          json.dumps({"action": "patch", "summary": "port", "edits": [{"file": "config.json", "find": '"alpha": 0.0', "replace": '"alpha": 0.0, "port": 1'}]})])
        r = f.fix("port", allow_prereg=True)
        self.assertTrue(r["needs_restart"]); self.assertTrue(f.status()["needs_restart"])
        f.restart()
        import time
        time.sleep(0.8)
        self.assertIn("RESTART", self.reloaded)

    def test_busy_and_empty(self):
        f, _ = self.make([])
        with self.assertRaises(ValueError):
            f.fix("   ")


class TestVerify(unittest.TestCase):
    def test_compile_step_on_real_file(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d).resolve()
            (root / "tests").mkdir()
            (root / "good.py").write_text("x = 1\n", "utf-8")
            (root / "bad.py").write_text("x = (\n", "utf-8")
            (root / "c.json").write_text("{", "utf-8")
            old = fixer.HERE
            fixer.HERE = root
            try:
                self.assertTrue(fixer.verify(["good.py"], run_tests=False)["ok"])
                r = fixer.verify(["bad.py"], run_tests=False)
                self.assertFalse(r["ok"]); self.assertIn("SyntaxError", r["steps"][0]["out"])
                self.assertFalse(fixer.verify(["c.json"], run_tests=False)["ok"])
                self.assertTrue(fixer.verify(["docs/x.md"], run_tests=True)["ok"])      # docs skip the tests
            finally:
                fixer.HERE = old


if __name__ == "__main__":
    unittest.main()

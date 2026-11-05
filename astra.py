"""Astra: the owner's OpenAI reviewer persona (model `gpt-6-astra`), here as two live helpers
for a validation session.

  LiveAnalyst   every `live_interval_s` it gets a compact snapshot of the streaming window
                (EEG flow index, band flow, pairs and their correlation, the protocol block,
                ratings, task accuracy, channel quality) plus what it already said, and writes
                a short observation. Notes go to the browser (SSE `astra`) and, while a session
                is recording, to sessions/<dir>/astra_notes.md + astra_notes.jsonl. `analyze()`
                runs a longer structured analysis against the pre-registered gates on demand;
                `ask()` answers a question with the same context.
  SetupHelper   a chat for electrode placement, montage, dongle / board switches, railed
                channels and band links. Every question carries the live setup snapshot
                (board, per-channel quality, raw peaks, frontal sites, band link), so "is
                channel 3 ok?" is answered from what the server sees right now.

Astra is observational: the system prompts forbid suggesting changes to the index weights,
artifact thresholds or verdict gates (pre-registration, docs/VALIDATION_PROTOCOL.md).

The key is read from the first `.env` above this folder that defines OPENAI_KEY
(BioProxal2026/.env; the variable name has a trailing space and the value single quotes in
the owner's file) on every call and is never logged or sent to the browser. Stdlib only:
urllib streaming of the chat-completions SSE, like firmware_v2/tools/astra_review.py.
"""
from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
API_URL = "https://api.openai.com/v1/chat/completions"
DEFAULT_MODEL = "gpt-6-astra"
NON_TASK = ("rest", "rest_ec", "pause")


# --- key + transport -----------------------------------------------------------------------
def find_env(start: Path = HERE, name: str = "OPENAI_KEY") -> Path | None:
    """Walk up from `start` to the first .env that mentions `name`."""
    for d in [start, *start.parents]:
        p = d / ".env"
        try:
            if p.exists() and name in p.read_text(encoding="utf-8", errors="replace"):
                return p
        except OSError:
            continue
    return None


def load_key(env_path: Path | None = None, name: str = "OPENAI_KEY") -> str:
    p = env_path or find_env()
    if p is None:
        raise RuntimeError("no .env with OPENAI_KEY found above " + str(HERE))
    for line in p.read_text(encoding="utf-8").splitlines():
        if "=" not in line:
            continue
        k, _, v = line.partition("=")
        if k.strip() != name:
            continue
        v = v.strip()
        if len(v) >= 2 and v[0] == v[-1] and v[0] in ("'", '"'):
            v = v[1:-1]
        if v:
            return v
    raise RuntimeError(f"{name} not set in {p}")


def parse_sse_chunk(line: bytes | str) -> str | None:
    """One `data: {...}` line of a chat-completions stream -> the delta text ('' when the
    chunk has none, None for non-data lines and the [DONE] sentinel)."""
    s = line.decode("utf-8", "replace") if isinstance(line, bytes) else line
    s = s.strip()
    if not s.startswith("data:"):
        return None
    s = s[5:].strip()
    if not s or s == "[DONE]":
        return None
    try:
        d = json.loads(s)
    except json.JSONDecodeError:
        return None
    try:
        delta = d["choices"][0].get("delta") or {}
        return delta.get("content") or ""
    except (KeyError, IndexError, AttributeError, TypeError):
        return ""


def chat(messages: list[dict], model: str = DEFAULT_MODEL, max_tokens: int = 1200, timeout: float = 180.0,
         on_delta=None, env_path: Path | None = None, retries: int = 2) -> str:
    """Chat completion, streamed when `on_delta` is given (called with each text piece).
    Returns the full text. 429 / 5xx / network errors are retried; other 4xx raise."""
    key = load_key(env_path)
    stream = on_delta is not None
    payload = {"model": model, "messages": messages, "max_completion_tokens": int(max_tokens)}
    if stream:
        payload["stream"] = True
    data = json.dumps(payload).encode("utf-8")
    last: Exception | None = None
    for attempt in range(retries + 1):
        req = urllib.request.Request(API_URL, data=data, method="POST",
                                     headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                if not stream:
                    body = json.loads(resp.read())
                    return body["choices"][0]["message"]["content"] or ""
                parts: list[str] = []
                for line in resp:
                    piece = parse_sse_chunk(line)
                    if piece:
                        parts.append(piece)
                        try:
                            on_delta(piece)
                        except Exception:  # noqa: BLE001
                            pass
                return "".join(parts)
        except urllib.error.HTTPError as e:
            last = e
            body = e.read().decode("utf-8", "replace")[:300]
            if e.code == 400 and stream and "stream" in body.lower():
                # model without streaming: one plain request instead
                return chat(messages, model, max_tokens, timeout, None, env_path, retries)
            if e.code != 429 and e.code < 500:
                raise RuntimeError(f"HTTP {e.code}: {body}") from e
            time.sleep(3 * (attempt + 1))
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            last = e
            time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"Astra call failed after {retries + 1} attempts: {last}")


# --- snapshots ------------------------------------------------------------------------------
def _f(v, d=2):
    return "-" if v is None else f"{float(v):.{d}f}"


def _pearson(xs, ys):
    n = len(xs)
    if n < 3:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx < 1e-12 or syy < 1e-12:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (sxx * syy) ** 0.5


def _clock(t):
    return datetime.fromtimestamp(t).strftime("%H:%M:%S")


def live_snapshot(pipeline, status: dict, window_s: float = 300.0, now: float | None = None) -> dict:
    """Everything the analyst needs about the last `window_s`, as plain numbers."""
    now = time.time() if now is None else now
    t0 = now - window_s
    eeg = [(t, v) for (t, v) in pipeline.eeg_hist if t >= t0 and v is not None]
    # 10-s means of the 2-s index so a 5-min window is ~30 numbers
    step, series = 10.0, []
    if eeg:
        k0 = int(eeg[0][0] // step)
        buckets: dict[int, list] = {}
        for t, v in eeg:
            buckets.setdefault(int(t // step) - k0, []).append(v)
        series = [round(sum(b) / len(b), 3) for _, b in sorted(buckets.items())]
    band = []
    for (t, fl, s) in pipeline.band_hist:
        if t >= t0:
            flags = [k for k in ("held", "settling", "asleep", "poor_contact") if s.get(k)]
            band.append({"t_rel": round(t - now), "flow": None if fl is None else round(fl, 3), "flags": flags,
                         "hr": s.get("hr")})
    pairs = [(t, b, e) for (t, b, e) in pipeline.pairs if t >= t0]
    r_win = _pearson([b for _, b, _ in pairs], [e for _, _, e in pairs]) if len(pairs) >= 3 else None
    st = pipeline.stats or {}
    blocks = (st.get("blocks") or {})
    proto = status.get("protocol") or {}
    blk = proto.get("block") or {}
    cur_idx = proto.get("block_index")
    trials = [tr for tr in pipeline.trials if cur_idx is not None and tr.get("block_index") == cur_idx]
    acc = (sum(1 for tr in trials if tr.get("correct")) / len(trials)) if trials else None
    marks = [{"t_rel": round(m["t"] - now), "kind": m["kind"], "value": m.get("value"), "note": m.get("note")}
             for m in pipeline.marks if m["t"] >= t0 and m["kind"] not in ("trial", "block_start", "block_end")]
    an = pipeline.last_analysis or {}
    q = an.get("quality") or []
    quality = [{"ch": i + 1, "label": pipeline.labels[i] if i < len(pipeline.labels) else f"ch{i + 1}",
                "state": qi.get("state"), "pct": qi.get("pct"), "enabled": bool(pipeline.enabled[i]) if i < len(pipeline.enabled) else True}
               for i, qi in enumerate(q)]
    fe = an.get("flow_eeg") or {}
    return {
        "now": round(now, 1), "clock": _clock(now), "window_s": window_s,
        "protocol": {"active": proto.get("active"), "name": proto.get("name"), "block_index": cur_idx,
                     "total_blocks": proto.get("total_blocks"), "condition": blk.get("condition"),
                     "task": blk.get("task"), "remaining_s": proto.get("remaining_s"),
                     "pending_rating": proto.get("pending_rating")},
        "flow_status": status.get("flow_index") or pipeline.flow_status(),
        "eeg_index_10s": series, "eeg_latest": fe.get("flow_eeg"), "eeg_artifact": fe.get("artifact"),
        "eeg_n": len(eeg),
        "band": band, "band_view": pipeline.band_view(), "band_link": status.get("band"),
        "pairs_window": [[round(t - now), round(b, 3), round(e, 3)] for (t, b, e) in pairs],
        "r_window": None if r_win is None else round(r_win, 3),
        "stats": {k: st.get(k) for k in ("n_pairs", "r", "r_ci", "r_last_5min", "mae", "band_minus_eeg_mean",
                                          "best_lag_s", "r_at_best_lag", "pair_window_s")},
        "blocks": blocks.get("rows") or [], "conditions": blocks.get("conditions") or [],
        "ordering": blocks.get("ordering") or {}, "block_rho": blocks.get("block_rho"),
        "n_blocks": blocks.get("n_blocks"),
        "task_block": {"trials": len(trials), "accuracy": None if acc is None else round(acc, 2),
                       "level": trials[-1].get("level") if trials else None},
        "marks": marks, "quality": quality,
        "recorder": status.get("recorder"),
    }


def render_live(s: dict) -> str:
    p = s["protocol"]
    fs = s.get("flow_status") or {}
    out = [f"LIVE WINDOW last {int(s['window_s'])} s, clock {s['clock']}"]
    if p.get("active"):
        out.append(f"protocol {p['name']}: block {p['block_index']}/{p['total_blocks']} condition={p['condition']} "
                   f"task={p['task']} remaining={_f(p['remaining_s'], 0)} s"
                   + (f" (rating pending for block {p['pending_rating']})" if p.get("pending_rating") is not None else ""))
    else:
        out.append("protocol: none running (free streaming)")
    out.append(f"EEG baseline {fs.get('baseline_state')} epochs={fs.get('epochs')} artifact_epochs={fs.get('artifact_epochs')} "
               f"frontal_sites={fs.get('frontal_sites')}")
    ser = s["eeg_index_10s"]
    if ser:
        out.append(f"EEG flow index, 10-s means oldest->newest (n_epochs={s['eeg_n']}): " + " ".join(_f(v) for v in ser)
                   + f" | latest {_f(s['eeg_latest'])}" + (" ARTIFACT" if s.get("eeg_artifact") else ""))
    else:
        out.append("EEG flow index: no clean epochs in window")
    b = s["band"]
    if b:
        out.append("band flow (t_rel s:flow flags): " + " ".join(
            f"{x['t_rel']}:{_f(x['flow'])}" + ("[" + ",".join(x['flags']) + "]" if x['flags'] else "") for x in b))
    else:
        out.append("band flow: no samples in window")
    bl = s.get("band_link") or {}
    bv = s.get("band_view") or {}
    out.append(f"band link {bl.get('source')} connected={bl.get('connected')} unit={bl.get('unit') or bv.get('band')} "
               f"last_sample_age={_f(bv.get('age_s'), 0)} s hr={bv.get('hr')} mode={bv.get('mode')} error={bl.get('error')}")
    if bv.get("spo2") is not None or bv.get("pi") is not None:
        out.append(f"band vitals: spo2={_f(bv.get('spo2'), 1)}% PI={_f(bv.get('pi'))}% RMSSD={_f(bv.get('rmssd'), 0)} ms "
                   f"SDNN={_f(bv.get('sdnn'), 0)} ms RR={_f(bv.get('rr'), 0)} base_hr={_f(bv.get('base_hr'), 0)} "
                   f"flow_raw={_f(bv.get('flow_raw'))} phase={bv.get('phase')} battery={bv.get('battery_mv')} mV")
    pw = s["pairs_window"]
    st = s["stats"]
    out.append(f"pairs in window n={len(pw)} r_window={_f(s['r_window'])}; session n_pairs={st.get('n_pairs')} r={_f(st.get('r'))} "
               f"CI={st.get('r_ci')} r_last_5min={_f(st.get('r_last_5min'))} MAE={_f(st.get('mae'))} "
               f"bias(band-eeg)={_f(st.get('band_minus_eeg_mean'))} best_lag={st.get('best_lag_s')} s r_lag={_f(st.get('r_at_best_lag'))}")
    if s["blocks"]:
        out.append("blocks so far (idx cond eeg_mean band_mean fss felt n_eeg n_band):")
        for r in s["blocks"]:
            out.append(f"  {r['index']} {r['condition']} {_f(r['eeg_mean'])} {_f(r['band_mean'])} {_f(r['fss_mean'])} "
                       f"{_f(r['felt'])} {r['n_eeg']} {r['n_band']}")
        o = s["ordering"]
        out.append(f"ordering eeg={o.get('eeg')} band={o.get('band')} fss={o.get('fss')}; block_rho={_f(s['block_rho'])} n_task_blocks={s['n_blocks']}")
    tb = s["task_block"]
    if tb["trials"]:
        out.append(f"task this block: trials={tb['trials']} accuracy={_f(tb['accuracy'])} level={tb['level']}")
    if s["marks"]:
        out.append("marks in window: " + "; ".join(f"{m['t_rel']}s {m['kind']}={m['value']}" + (f' "{m["note"]}"' if m.get('note') else '') for m in s["marks"]))
    if s["quality"]:
        out.append("channels: " + " | ".join(f"{q['ch']} {q['label']} {q['state']} {_f(q['pct'], 0)}%" + ("" if q['enabled'] else " OFF") for q in s["quality"]))
    rec = s.get("recorder") or {}
    out.append(f"recording={'yes ' + str(rec.get('dir')) if rec.get('active') else 'no'}")
    return "\n".join(out)


def setup_snapshot(pipeline, status: dict, raw_peak=None) -> dict:
    an = pipeline.last_analysis or {}
    q = an.get("quality") or []
    chans = []
    for i in range(pipeline.n_ch):
        qi = q[i] if i < len(q) else {}
        chans.append({"ch": i + 1, "label": pipeline.labels[i] if i < len(pipeline.labels) else f"ch{i + 1}",
                      "state": qi.get("state", "unknown"), "pct_full_scale": qi.get("pct"), "std_uv": qi.get("std"),
                      "raw_peak_uv": None if raw_peak is None or i >= len(raw_peak) else round(float(raw_peak[i]), 1),
                      "enabled": bool(pipeline.enabled[i]) if i < len(pipeline.enabled) else True})
    bands = an.get("bands") or {}
    rel = bands.get("rel") or {}
    theta = rel.get("theta")
    alpha = rel.get("alpha")
    fs = status.get("flow_index") or pipeline.flow_status()
    return {"clock": _clock(time.time()), "board": status.get("board"), "filter": status.get("filter"),
            "montage": list(pipeline.labels), "channels": chans,
            "rel_theta": theta, "rel_alpha": alpha,
            "frontal_sites": fs.get("frontal_sites"), "baseline_state": fs.get("baseline_state"),
            "band_link": status.get("band"), "band_view": pipeline.band_view(),
            "recorder": status.get("recorder"),
            "protocol_available": [a.get("name") for a in ((status.get("protocol") or {}).get("available") or [])]}


def render_setup(s: dict) -> str:
    b = s.get("board") or {}
    out = [f"SETUP STATE clock {s['clock']}",
           f"board kind={b.get('kind')} port={b.get('port')} n_ch={b.get('n_ch')} fs={b.get('fs')} daisy={b.get('daisy')} "
           f"streaming={b.get('streaming')} packets={b.get('packets')} dropped={b.get('dropped')} last_packet_age={_f(b.get('last_packet_age_s'), 1)} s",
           f"display filter {s.get('filter')}", f"montage labels (ch1..): {' '.join(s['montage'])}",
           f"frontal sites used by the index: {s.get('frontal_sites')}; baseline {s.get('baseline_state')}",
           "channels (ch label quality pct_of_full_scale std_uV raw_peak_uV rel_theta rel_alpha):"]
    th, al = s.get("rel_theta") or [], s.get("rel_alpha") or []
    for i, c in enumerate(s["channels"]):
        out.append(f"  {c['ch']} {c['label']} {c['state']} {_f(c['pct_full_scale'], 1)}% {_f(c['std_uv'], 1)} {_f(c['raw_peak_uv'], 0)} "
                   f"{_f(th[i] if i < len(th) else None)} {_f(al[i] if i < len(al) else None)}" + ("" if c['enabled'] else " (disabled)"))
    bl = s.get("band_link") or {}
    bv = s.get("band_view") or {}
    out.append(f"band link source={bl.get('source')} connected={bl.get('connected')} unit={bl.get('unit') or bv.get('band')} "
               f"error={bl.get('error')} seen={bl.get('seen')} last_sample_age={_f(bv.get('age_s'), 0)} s flow={_f(bv.get('flow'))} "
               f"flags held={bv.get('held')} settling={bv.get('settling')} asleep={bv.get('asleep')} poor_contact={bv.get('poor_contact')}")
    rec = s.get("recorder") or {}
    out.append(f"recording={'yes' if rec.get('active') else 'no'}; protocols: {', '.join(s.get('protocol_available') or [])}")
    return "\n".join(out)


# --- prompts ----------------------------------------------------------------------------------
ANALYST_SYSTEM = """You are Astra, the live analyst sitting beside a pre-registered validation session: an OpenBCI Cyton (Ultracortex) EEG flow index is the reference, and the BioProxal wrist band's flow score (0-1) is the thing under test.

What you know:
- EEG index = frontal-midline theta (4-7 Hz) relative power, log, z-scored against the wearer's own eyes-open rest block, through a sigmoid: 0.5 = at rest, 0.73 = one resting SD above. It only means something after the baseline is `rest-frozen`; before that it is `session-until-rest` and drifts.
- Protocol `difficulty_ladder`: 3-min rest (baseline), 1-min eyes closed, then two rounds of boredom / flow / overload 6-min arithmetic blocks with a Flow Short Scale (FSS, 1-7) rating and 60-s pause after each, 2-min rest. Expected: frontal theta and the band both highest in the `flow` blocks; FSS flow > boredom and flow > overload (manipulation check).
- Pre-registered gates, judged per block at the end: G1 manipulation check (FSS), G2 band mean highest in flow, G3 Spearman rho between band and EEG block means >= 0.5 over >= 6 task blocks. Epoch-level r in the window is secondary and noisy (the band updates once a minute on Supabase; pairs are band samples vs the mean EEG index over the pair window).
- Band flags held / settling / asleep / poor_contact mean that sample is excluded.

Your job every tick: read the snapshot, compare it with what you said before, and write ONE short note (max 80 words). Start with a tag: [OK], [WATCH] or [ISSUE]. Report: data-quality problems (railed/near/flat channels, artifact epochs rising, band link stale or flagged, baseline not frozen), whether the current block's EEG index and band flow are moving the way the condition predicts, and the running agreement (r_window, block means, block_rho). Say "no change" when nothing new happened. State numbers, do not speculate about mechanisms.

Rules: observational only. Never propose changing index weights, artifact thresholds, pair windows, gates or the schedule mid-session (pre-registration). Do not praise. No markdown headings; plain sentences."""

ANALYSIS_SYSTEM = ANALYST_SYSTEM + """

For this request write a structured interim analysis (max 300 words) with these headings in plain text lines: "Data quality", "Blocks so far", "Agreement", "Toward the gates" (what G1/G2/G3 would say if the session ended now and what is still missing), "Watch next". Numbers from the snapshot only."""

SETUP_SYSTEM = """You are Astra, the setup coach for a flow-validation rig: OpenBCI Cyton v3 (8 ch, or 16 with Daisy) on an Ultracortex Mark IV headset, RFduino USB dongle, plus a BioProxal wrist band whose flow score arrives through Supabase (phone app), direct BLE, USB or a simulator. A localhost dashboard streams everything; with every question you get its live SETUP STATE (board, per-channel quality, montage, band link). Answer from that state first, then from the reference below. Be concrete: name channels as "ch3 (F3)", give the physical action, and say what the dashboard should show when it is fixed. Keep answers under 150 words unless the user asks for detail. No praise, no filler.

REFERENCE
Electrode placement (10-20): Fz = midline, 30% of the nasion-inion distance back from the nasion (one hand-width above the brow line, centred). Cz = vertex, midway nasion-inion and midway ear-to-ear. F3/F4 = roughly 5 cm left/right of Fz and slightly forward (above the mid-eyebrow, hairline-plus-two-fingers). C3/C4 = 7 cm either side of Cz on the ear-to-ear line. P3/P4 = mirror of F3/F4 behind Cz. Fp1/Fp2 = forehead above each eye (factory sites; blink-dominated). O1/O2 = back of the head above the inion. P7/P8 = low parietal above the ears, behind T7/T8.
Flow montage for this study (start the app with --montage flow): ch1 Fz, ch2 F3, ch3 F4, ch4 Cz, ch5 C3, ch6 C4, ch7 P3, ch8 P4. The index reads theta at Fz/F3/F4 (frontal_sites should show Fz,F3,F4); Cz/C3/C4/P3/P4 give the alpha covariate. Factory montage: Fp1 Fp2 C3 C4 P7 P8 O1 O2 (frontal_sites Fp1,Fp2 - acceptable but label the session). 16-ch Daisy frontal set is F7 F8 F3 F4.
Ultracortex Mark IV: spiky dry comb electrodes screw into the numbered node holes; turn each until it touches scalp, then a quarter turn more; part hair under the combs. Flat snap electrodes for forehead sites. Ear clips: SRB2 (reference, bottom pin pair) and BIAS (ground) both on earlobes - without them every channel reads railed. Hex standoffs tighten the frame; the headset should sit level with Cz at the vertex.
Cyton switches: board switch to PC, dongle switch to GPIO6, board blue LED on. One dongle is paired to one board. Close the OpenBCI GUI before starting this app (one serial owner).
Quality flags (raw data): railed = |x| > 90% of full scale (electrode off scalp, ear clip off, loose wire), near = > 75%, flat = no variation (open input / unplugged / channel off on the board), good = otherwise. Expect raw peaks of tens of microvolts with eyes open; hundreds mean movement, jaw clench or poor contact; thousands mean rail. Blinks make 100-200 uV bumps on Fp/Fz/F3/F4 - fine during tasks, the artifact rule drops them, but ask the wearer to sit still during the 3-min rest baseline.
Mains: a 60 Hz FFT peak with the notch off is normal; set notch 60 (50 in a 50-Hz country). The display band-pass never changes the index.
Alpha check: eyes closed 20 s -> alpha (8-13 Hz) rises at P3/P4/O1/O2. Theta check: the index needs the rest baseline first; frontal theta cannot be judged from one epoch.
Daisy: if tapping the electrode labelled F3 (ch11) moves a trace in ch1-8, flip daisy_even_is_upper in config.json.
Band: Supabase source needs the phone app connected and uploading (one row per minute, wearer picked in the band panel, age under 120 s). BLE source needs the phone app closed (one BLE connection). USB needs the cable, the firmware tools dir and the master key. Band flags: held = flow frozen (mode 1), settling = first minutes after putting it on, asleep, poor_contact = loose strap - a band sample with any flag is excluded from pairs. The band should be worn at least 10 min before the protocol starts.
Session flow: Start stream -> all channels good -> notch ok -> pick wearer / band connected -> START PROTOCOL (recording starts itself) -> follow the screen -> stop recording -> Sessions -> Report.
"""


# --- helpers ----------------------------------------------------------------------------------
class LiveAnalyst:
    def __init__(self, pipeline, status_fn, cfg: dict, emit, log=print, chat_fn=chat, recorder=None):
        self.pipeline, self.status_fn, self.cfg, self.emit, self.log = pipeline, status_fn, cfg, emit, log
        self.chat_fn = chat_fn
        self.recorder = recorder
        self.model = str(cfg.get("model", DEFAULT_MODEL))
        self.interval_s = float(cfg.get("live_interval_s", 60))
        self.window_s = float(cfg.get("window_min", 5)) * 60.0
        self.max_tokens = int(cfg.get("max_tokens", 1200))
        self.keep_turns = int(cfg.get("memory_turns", 8))
        self.enabled = False
        self.notes: list[dict] = []           # {id, t, kind, text, block, condition}
        self.history: list[dict] = []         # alternating user/assistant, last keep_turns pairs
        self.busy = False
        self.error: str | None = None
        self.last_tick = 0.0
        self._lock = threading.Lock()
        self._id = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="astra-live", daemon=True)
        self._thread.start()

    # --- control ---
    def set_live(self, enabled: bool, interval_s: float | None = None) -> None:
        if interval_s is not None:
            self.interval_s = max(20.0, float(interval_s))
        self.enabled = bool(enabled)
        if self.enabled:
            self.last_tick = 0.0          # tick now
        self.log(f"[astra] live analyst {'on' if self.enabled else 'off'} every {self.interval_s:.0f} s")

    def _run(self) -> None:
        while not self._stop.is_set():
            self._stop.wait(1.0)
            if not self.enabled or self.busy:
                continue
            if time.time() - self.last_tick >= self.interval_s:
                self.last_tick = time.time()
                try:
                    self.tick()
                except Exception as e:  # noqa: BLE001
                    self.error = str(e)
                    self.log(f"[astra] tick failed: {e}")
                    self.emit({"kind": "error", "role": "analyst", "text": str(e), "t": time.time()})

    def stop(self) -> None:
        self._stop.set()

    # --- one exchange ---
    def _next_id(self) -> str:
        self._id += 1
        return f"a{int(time.time())}-{self._id}"

    def _request(self, kind: str, system: str, user_text: str, remember: bool, extra: dict | None = None) -> dict:
        with self._lock:
            self.busy = True
            nid = self._next_id()
            t = time.time()
            snap = self._snapshot()
            extra = extra or {}
            msgs = [{"role": "system", "content": system}, *self.history, {"role": "user", "content": user_text}]
            self.emit({"kind": kind, "role": "analyst", "id": nid, "t": t, "text": "", "done": False, **extra,
                       "block": snap["protocol"].get("block_index"), "condition": snap["protocol"].get("condition")})
            try:
                text = self.chat_fn(msgs, model=self.model, max_tokens=self.max_tokens,
                                    on_delta=lambda piece: self.emit({"kind": "delta", "id": nid, "text": piece}))
                self.error = None
            except Exception as e:  # noqa: BLE001
                self.busy = False
                self.error = str(e)
                self.emit({"kind": "error", "role": "analyst", "id": nid, "t": t, "text": str(e), "done": True})
                raise
            note = {"id": nid, "t": t, "kind": kind, "text": text.strip(), "block": snap["protocol"].get("block_index"),
                    "condition": snap["protocol"].get("condition"), **extra}
            self.notes.append(note)
            if len(self.notes) > 500:
                del self.notes[: len(self.notes) - 500]
            if remember:
                self.history.extend([{"role": "user", "content": user_text}, {"role": "assistant", "content": note["text"]}])
                if len(self.history) > 2 * self.keep_turns:
                    del self.history[: len(self.history) - 2 * self.keep_turns]
            self._write(note)
            self.emit({**note, "role": "analyst", "done": True})
            self.busy = False
            return note

    def _snapshot(self) -> dict:
        return live_snapshot(self.pipeline, self.status_fn(), self.window_s)

    def tick(self) -> dict:
        snap = self._snapshot()
        return self._request("note", ANALYST_SYSTEM, render_live(snap) + "\n\nWrite your note.", remember=True)

    def analyze(self) -> dict:
        snap = live_snapshot(self.pipeline, self.status_fn(), max(self.window_s, 900.0))
        return self._request("analysis", ANALYSIS_SYSTEM, render_live(snap) + "\n\nWrite the interim analysis.", remember=True)

    def ask(self, question: str) -> dict:
        snap = self._snapshot()
        q = str(question).strip()[:2000]
        return self._request("reply", ANALYST_SYSTEM + "\n\nThe operator asks a question; answer it directly (max 150 words) from the snapshot and your earlier notes.",
                             render_live(snap) + f"\n\nOPERATOR QUESTION: {q}", remember=True, extra={"question": q})

    # --- persistence ---
    def _write(self, note: dict) -> None:
        rec = self.recorder
        if rec is None or not getattr(rec, "active", False) or not getattr(rec, "dir", None):
            return
        try:
            d = Path(rec.dir)
            stamp = _clock(note["t"])
            tag = {"note": "", "analysis": "ANALYSIS ", "reply": "Q&A "}.get(note["kind"], "")
            where = f" block {note['block']} {note['condition']}" if note.get("block") is not None else ""
            q = f"Q: {note['question']} -- " if note.get("question") else ""
            body = note["text"].replace("\n", "\n  ")
            with open(d / "astra_notes.md", "a", encoding="utf-8") as f:
                if f.tell() == 0:
                    f.write(f"# Astra live notes ({self.model})\n\n")
                f.write(f"- **{stamp}**{where} {tag}{q}{body}\n")
            with open(d / "astra_notes.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(note) + "\n")
        except OSError as e:
            self.log(f"[astra] could not write notes: {e}")

    def status(self) -> dict:
        return {"enabled": self.enabled, "interval_s": self.interval_s, "window_s": self.window_s, "busy": self.busy,
                "model": self.model, "n_notes": len(self.notes), "error": self.error,
                "next_in_s": None if not self.enabled else max(0.0, round(self.interval_s - (time.time() - self.last_tick), 1))}


class SetupHelper:
    def __init__(self, pipeline, status_fn, cfg: dict, emit, log=print, chat_fn=chat, raw_peak_fn=None, reference: str = ""):
        self.pipeline, self.status_fn, self.cfg, self.emit, self.log = pipeline, status_fn, cfg, emit, log
        self.chat_fn = chat_fn
        self.raw_peak_fn = raw_peak_fn or (lambda: None)
        self.model = str(cfg.get("model", DEFAULT_MODEL))
        self.max_tokens = int(cfg.get("max_tokens", 1200))
        self.keep_turns = int(cfg.get("memory_turns", 8))
        self.system = SETUP_SYSTEM + ("\n\nPROJECT DOCS\n" + reference if reference else "")
        self.transcript: list[dict] = []      # {id, t, who: user|astra, text}
        self.history: list[dict] = []
        self.busy = False
        self.error: str | None = None
        self._lock = threading.Lock()
        self._id = 0

    def snapshot(self) -> dict:
        return setup_snapshot(self.pipeline, self.status_fn(), self.raw_peak_fn())

    def clear(self) -> None:
        with self._lock:
            self.transcript, self.history = [], []

    def ask(self, question: str) -> dict:
        q = str(question).strip()[:2000]
        if not q:
            raise ValueError("empty question")
        with self._lock:
            self.busy = True
            self._id += 1
            nid = f"s{int(time.time())}-{self._id}"
            t = time.time()
            self.transcript.append({"id": nid + "u", "t": t, "who": "user", "text": q})
            state = render_setup(self.snapshot())
            user = f"{state}\n\nQUESTION: {q}"
            msgs = [{"role": "system", "content": self.system}, *self.history, {"role": "user", "content": user}]
            self.emit({"kind": "setup", "id": nid, "t": t, "question": q, "text": "", "done": False})
            try:
                text = self.chat_fn(msgs, model=self.model, max_tokens=self.max_tokens,
                                    on_delta=lambda piece: self.emit({"kind": "delta", "id": nid, "text": piece}))
                self.error = None
            except Exception as e:  # noqa: BLE001
                self.busy = False
                self.error = str(e)
                self.emit({"kind": "error", "role": "setup", "id": nid, "t": t, "text": str(e), "done": True})
                raise
            text = text.strip()
            self.transcript.append({"id": nid, "t": t, "who": "astra", "text": text})
            # remember the question without the state dump, so history stays small and current
            self.history.extend([{"role": "user", "content": q}, {"role": "assistant", "content": text}])
            if len(self.history) > 2 * self.keep_turns:
                del self.history[: len(self.history) - 2 * self.keep_turns]
            if len(self.transcript) > 200:
                del self.transcript[: len(self.transcript) - 200]
            self.emit({"kind": "setup", "id": nid, "t": t, "question": q, "text": text, "done": True})
            self.busy = False
            return {"id": nid, "t": t, "text": text}

    def check(self) -> dict:
        return self.ask("Check my setup: go channel by channel, then the ear clips / mains / band link, and list what to "
                        "fix before I start the protocol. End with READY or NOT READY.")

    def status(self) -> dict:
        return {"busy": self.busy, "model": self.model, "n_turns": len(self.transcript) // 2, "error": self.error}


def load_reference(root: Path = HERE) -> str:
    """README troubleshooting + the manual hardware checks, for the setup helper's system prompt."""
    parts = []
    readme = root / "README.md"
    if readme.exists():
        txt = readme.read_text("utf-8", errors="replace")
        for head in ("## Montage", "## Band sources", "## Troubleshooting"):
            i = txt.find(head)
            if i >= 0:
                j = txt.find("\n## ", i + 3)
                parts.append(txt[i: j if j > 0 else len(txt)].strip())
    tp = root / "docs" / "TEST_PLAN.md"
    if tp.exists():
        txt = tp.read_text("utf-8", errors="replace")
        i = txt.find("## 3. Manual hardware checks")
        if i >= 0:
            j = txt.find("\n## ", i + 3)
            parts.append(txt[i: j if j > 0 else len(txt)].strip())
    return "\n\n".join(parts)[:16000]


class Astra:
    """Facade wired by app.py; the server calls `command` and `status`."""

    def __init__(self, pipeline, status_fn, cfg: dict, broadcast, log=print, recorder=None, chat_fn=chat,
                 raw_peak_fn=None, reference: str | None = None, log_tail_fn=None, roots_fn=None, restart_fn=None,
                 protocol_reload_fn=None, post_reload_fn=None, fixer=None):
        self.cfg = cfg or {}
        self.log = log
        self.pipeline, self.status_fn = pipeline, status_fn
        self.available = find_env() is not None
        emit = lambda d: broadcast("astra", d)  # noqa: E731
        self.analyst = LiveAnalyst(pipeline, status_fn, self.cfg, emit, log=log, chat_fn=chat_fn, recorder=recorder)
        self.setup = SetupHelper(pipeline, status_fn, self.cfg, emit, log=log, chat_fn=chat_fn, raw_peak_fn=raw_peak_fn,
                                 reference=load_reference() if reference is None else reference)
        self.log_tail_fn = log_tail_fn or (lambda n: [])
        self.roots_fn = roots_fn or (lambda: [])
        self.post_reload_fn = post_reload_fn
        if fixer is None:
            import fixer as _fixer

            def fix_chat(msgs, model, max_tokens):
                try:
                    return chat_fn(msgs, model=model, max_tokens=max_tokens, timeout=600.0)
                except TypeError:
                    return chat_fn(msgs, model=model, max_tokens=max_tokens)
            fixer = _fixer.Fixer(self.cfg, emit, fix_chat, self.fix_context, self.reload_module, log=log,
                                 restart_fn=restart_fn, protocol_reload_fn=protocol_reload_fn)
        self.fixer = fixer
        if not self.available:
            log("[astra] no .env with OPENAI_KEY above this folder; Astra panels will show 'no key'")
        elif self.cfg.get("auto_live"):
            self.analyst.set_live(True)

    # --- for the fixer ---
    def fix_context(self) -> str:
        parts = []
        try:
            st = self.status_fn()
            parts.append(render_live(live_snapshot(self.pipeline, st, 300.0)))
            parts.append(render_setup(setup_snapshot(self.pipeline, st, getattr(self.pipeline, "last_raw_peak", None))))
        except Exception as e:  # noqa: BLE001
            parts.append(f"(snapshot failed: {e})")
        tail = self.log_tail_fn(40)
        if tail:
            parts.append("SERVER LOG (last lines):\n" + "\n".join(str(x) for x in tail))
        if self.analyst.notes:
            parts.append("ASTRA'S RECENT NOTES:\n" + "\n".join(f"- {n['text']}" for n in self.analyst.notes[-5:]))
        return "\n\n".join(parts)

    def reload_module(self, name: str) -> dict:
        import hotreload
        info = hotreload.reload_module(name, self.roots_fn(), HERE, log=self.log)
        if self.post_reload_fn:
            self.post_reload_fn(name, info)
        return info

    def status(self) -> dict:
        return {"available": self.available, "model": str(self.cfg.get("model", DEFAULT_MODEL)),
                "live": self.analyst.status(), "setup": self.setup.status(), "fix": self.fixer.status()}

    def state(self) -> dict:
        return {**self.status(), "notes": self.analyst.notes[-200:], "setup_transcript": self.setup.transcript[-60:],
                "fixes": self.fixer.fixes[-50:]}

    def _bg(self, fn, *a):
        threading.Thread(target=self._guard, args=(fn, *a), daemon=True).start()

    def _guard(self, fn, *a):
        try:
            fn(*a)
        except Exception as e:  # noqa: BLE001
            self.log(f"[astra] {getattr(fn, '__name__', 'call')} failed: {e}")

    def command(self, action: str, body: dict) -> dict:
        if not self.available and action not in ("astra_live",):
            raise ValueError("Astra has no key: put OPENAI_KEY in BioProxal2026/.env")
        if action == "astra_live":
            if not self.available and body.get("enabled"):
                raise ValueError("Astra has no key: put OPENAI_KEY in BioProxal2026/.env")
            self.analyst.set_live(bool(body.get("enabled", True)), body.get("interval_s"))
            return {"astra": self.status()}
        if action == "astra_analyze":
            if self.analyst.busy:
                raise ValueError("Astra is still writing")
            self._bg(self.analyst.analyze)
            return {"started": True}
        if action == "astra_ask":
            text = str(body.get("text") or "").strip()
            if not text:
                raise ValueError("empty question")
            role = str(body.get("role") or "analyst")
            if role == "setup":
                if self.setup.busy:
                    raise ValueError("Astra is still answering")
                self._bg(self.setup.ask, text)
            else:
                if self.analyst.busy:
                    raise ValueError("Astra is still writing")
                self._bg(self.analyst.ask, text)
            return {"started": True}
        if action == "astra_check":
            if self.setup.busy:
                raise ValueError("Astra is still answering")
            self._bg(self.setup.check)
            return {"started": True}
        if action == "astra_clear":
            self.setup.clear()
            return {"cleared": True}
        if action == "astra_fix":
            text = str(body.get("text") or "").strip()
            if not text:
                raise ValueError("describe the issue")
            if self.fixer.busy:
                raise ValueError("Astra is still working on a fix")
            self._bg(self.fixer.fix, text, bool(body.get("allow_prereg")))
            return {"started": True}
        if action == "astra_revert":
            if self.fixer.busy:
                raise ValueError("Astra is still working on a fix")
            return self.fixer.revert()
        if action == "astra_restart":
            st = self.status_fn()
            proto = st.get("protocol") or {}
            rec = st.get("recorder") or {}
            if (proto.get("active") or rec.get("active")) and not body.get("force"):
                raise ValueError("a protocol or recording is running; stop it first (or force)")
            return self.fixer.restart()
        raise ValueError(f"unknown action {action!r}")

#!/usr/bin/env python3
"""v4 MiniWoB sweep runner — entry J STAGE 2, PRE-REGISTERED (IMPLEMENTATION_PLAN.md).

Mechanics identical to the entry-E v3 runner (run_v3.py) except the four
frozen v4 deltas: a fresh /dev/urandom seed RNG (3002643678, drawn
2026-09-24 before any session), exclusion of every v1/v2/v3 seed, fresh
output paths (v4), and the throwaway sanity seed 3306495589. Task list,
prompt, arm, model string (poc-proxy-12b/gemma-12b-decide), caps, and
CDP collection are verbatim run_v3.

  python3 run_v4.py plan              # print the frozen seed plan and exit
  python3 run_v4.py sanity            # one throwaway session -> sanity_runs_v4/
  python3 run_v4.py sweep             # the frozen 168-session sweep, resumable
"""
import json
import os
import random
import subprocess
import sys
import threading
import time
import urllib.request

import websocket  # websocket-client

PAGE_BASE = "http://127.0.0.1:8077/miniwob"
ARM = "chain"
MODEL = "poc-proxy-12b/gemma-12b-decide"
OMP_BIN = os.path.expanduser("~/.bun/bin/omp")
OMP_CWD = "/home/sahal/workspace/refleqwen"
RUNS = "/tmp/mw_bench/main_runs_v4"
SANITY = "/tmp/mw_bench/sanity_runs_v4"
MAIN = "/tmp/mw_bench/main_v4.jsonl"
CDP_MAIN = "/tmp/mw_bench/main_v4.jsonl.cdp"
SEED_RNG = 3002643678
N_SEEDS = 7
OMP_CAP_S = 480
SETTLE_S = 10
PAGE_CAP_S = 120

TASKS = sorted([
    "click-button", "click-button-sequence", "click-checkboxes",
    "click-collapsible", "click-dialog", "click-link", "click-option",
    "click-tab", "click-tab-2", "click-test", "click-widget", "enter-date",
    "enter-password", "enter-text", "enter-text-2", "focus-text",
    "focus-text-2", "form-sequence", "login-user", "login-user-popup",
    "navigate-tree", "read-table", "search-engine", "use-autocomplete",
])

PROMPT = (
    "Browser task. In the eval tool, use the browser primitives like this: "
    "`const tab = await browser.open({{name:'w{seed}c0', url:'{url}'}}); "
    "display(await tab.ariaSnapshot());` and in later cells re-acquire with "
    "`const tab = await browser.tab('w{seed}c0');`. To act, write "
    "self-contained JS in `await tab.evaluate(` + backticks: find the "
    "element by its exact visible text or accessible name "
    "(document.querySelectorAll over a, button, [role=button], input, "
    "select, textarea, label) and call .click() on it; for text fields set "
    ".value with the native setter and dispatch input and change events; "
    "for <select> set the option then dispatch change. Verify each action "
    "with `display(await tab.ariaSnapshot());`. The goal is stated on the "
    "page itself, in the instruction area at the top of the page. Do what "
    "it says, then report done with evidence from the last snapshot. Work "
    "briskly; the episode is capped at 120 seconds."
)

STATUS_JS = (
    "JSON.stringify({done: (typeof WOB_DONE_GLOBAL==='undefined'?null:"
    "WOB_DONE_GLOBAL), raw: (typeof WOB_RAW_REWARD_GLOBAL==='undefined'?null:"
    "WOB_RAW_REWARD_GLOBAL), fin: (typeof WOB_REWARD_GLOBAL==='undefined'?null:"
    "WOB_REWARD_GLOBAL), reason: (typeof WOB_REWARD_REASON==='undefined'?null:"
    "WOB_REWARD_REASON), ep: (typeof WOB_EPISODE_ID==='undefined'?null:"
    "WOB_EPISODE_ID), ready: (typeof WOB_TASK_READY==='undefined'?null:"
    "WOB_TASK_READY), query: ((document.getElementById('query')||{})"
    ".textContent||'').replace(/\\s+/g,' ').trim()})"
)


def existing_seeds():
    out = set()
    for f in ("/tmp/mw_bench/main.jsonl", "/tmp/mw_bench/main_v2.jsonl",
              "/tmp/mw_bench/main_v3.jsonl"):
        for line in open(f):
            out.add(int(json.loads(line)["seed"]))
    return out


def seed_plan():
    rng = random.Random(SEED_RNG)
    taken = existing_seeds()
    plan = []
    for t in TASKS:
        col = []
        while len(col) < N_SEEDS:
            s = rng.randint(10 ** 9, 10 ** 10 - 1)
            if s in taken or s in col:
                continue
            taken.add(s)
            col.append(s)
        plan.append((t, col))
    return plan


def chrome_snapshot():
    out = subprocess.run(["pgrep", "-af", "remote-debugging-port=0"],
                         capture_output=True, text=True).stdout
    return {l.split(" ", 1)[0]: l for l in out.strip().splitlines() if l}


def udd_of(line):
    for tok in line.split():
        if tok.startswith("--user-data-dir="):
            return tok.split("=", 1)[1]
    return None


def candidate_ports(since_pids):
    """All DevTools ports from chrome processes, new pids first."""
    ports = []
    for pid, line in chrome_snapshot().items():
        udd = udd_of(line)
        if not udd:
            continue
        fp = os.path.join(udd, "DevToolsActivePort")
        if os.path.exists(fp):
            try:
                p = int(open(fp).readline().strip())
                ports.append((0 if pid not in since_pids else 1, p))
            except Exception:
                pass
    ports.sort()
    return list(dict.fromkeys(p for _, p in ports))


def list_targets(port):
    try:
        with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/json", timeout=3) as r:
            return json.load(r)
    except Exception:
        return []


def eval_on(ws, expr, mid=1):
    ws.send(json.dumps({"id": mid, "method": "Runtime.evaluate",
                        "params": {"expression": expr, "returnByValue": True}}))
    while True:
        msg = json.loads(ws.recv())
        if msg.get("id") == mid:
            return msg.get("result", {}).get("result", {}).get("value")


def collect(task, seed, t0, out, dbg=None):
    """Attach to the agent's tab, poll page state until the episode ends.

    v2 mechanics: the collector runs concurrently with omp, attaches as
    soon as the agent's Chromium appears, polls the WOB_* globals until
    done, lets the reward settle ~6 s, and records the final state. After
    omp exits it keeps polling only for the settle grace in out["cdp_stop"].
    """

    def log(msg):
        if dbg:
            with open(dbg, "a") as f:
                f.write(f"[{time.time()-t0:7.1f}] {msg}\n")

    try:
        deadline = t0 + OMP_CAP_S + SETTLE_S
        port = None
        first = final = None
        ws = None
        cur_ws_url = None
        log(f"collect start, since_pids n={len(out['since_pids'])}")
        while time.time() < deadline:
            stop = out.get("cdp_stop")
            if stop is not None and first is None and time.time() > stop:
                # before attach: omp is gone and no browser ever appeared.
                # after attach the deadline is pegged to the page cap below.
                log("stop grace exceeded")
                break
            if port is None:
                # tab-driven discovery: the session's (task, seed) URL is
                # unique, so scan every candidate DevTools port for it
                for cand in candidate_ports(out["since_pids"]):
                    tgt = [t for t in list_targets(cand)
                           if t.get("type") == "page"
                           and f"/{task}.html" in t.get("url", "")
                           and f"wseed={seed}" in t.get("url", "")]
                    if tgt:
                        port = cand
                        break
                log(f"port scan -> {port}")
                if port is None:
                    time.sleep(1)
                continue
            tgt = [t for t in list_targets(port)
                   if t.get("type") == "page"
                   and f"/{task}.html" in t.get("url", "")
                   and f"wseed={seed}" in t.get("url", "")]
            if not tgt:
                time.sleep(1)
                continue
            url = tgt[0].get("webSocketDebuggerUrl")
            if url != cur_ws_url:
                try:
                    if ws:
                        ws.close()
                    ws = websocket.create_connection(url, timeout=5,
                                                     suppress_origin=True)
                    cur_ws_url = url
                    log(f"ws attached: {url[:60]}")
                except Exception as e:
                    cur_ws_url = None
                    log(f"ws attach FAIL: {e}")
                    time.sleep(1)
                    continue
            try:
                val = eval_on(ws, STATUS_JS)
                st = json.loads(val) if val else None
            except Exception as e:
                cur_ws_url = None
                log(f"eval FAIL: {type(e).__name__} {e}")
                time.sleep(1)
                continue
            if st is None:
                time.sleep(1)
                continue
            st["time_left"] = not st.get("done")
            if first is None:
                first = st
                out["port"] = port
                # the page episode runs PAGE_CAP_S from open regardless of
                # when omp exits: keep polling until it can end + settle
                deadline = max(deadline, min(time.time() + PAGE_CAP_S + 12,
                                             t0 + OMP_CAP_S + SETTLE_S + 40))
                log(f"first state: raw={st.get('raw')} done={st.get('done')} q={str(st.get('query'))[:40]!r}")
            final = st
            if st.get("done"):
                time.sleep(6)  # let WOB_REWARD_GLOBAL settle
                out["waited_s"] = 6.0
                try:
                    val = eval_on(ws, STATUS_JS)
                    if val:
                        final = json.loads(val)
                        final["time_left"] = not final.get("done")
                except Exception:
                    pass
                log(f"DONE: raw={final.get('raw')} fin={final.get('fin')} reason={final.get('reason')}")
                break
            time.sleep(1.5)
        if final is not None and not final.get("done"):
            out["timed_out_wait"] = True
            out["waited_s"] = round(time.time() - t0, 1)
        if ws:
            try:
                ws.close()
            except Exception:
                pass
        out["first"] = first
        out["final"] = final
        log("collect end")
    except Exception:
        import traceback
        log("COLLECT CRASH:\n" + traceback.format_exc())


def run_session(task, seed, runs_dir):
    tag = f"{ARM}/{task}/s{seed}"
    done_tags = set()
    if os.path.exists(MAIN):
        for line in open(MAIN):
            done_tags.add(json.loads(line)["tag"])
    if tag in done_tags:
        return None

    sdir = os.path.join(runs_dir, f"{task}_s{seed}_{ARM}")
    os.makedirs(sdir, exist_ok=True)
    url = f"{PAGE_BASE}/{task}.html?wseed={seed}&autostart=1&rep=c"
    prompt = PROMPT.format(seed=seed, url=url)

    out = {"tag": tag, "match": f"{task}.html?wseed={seed}&autostart=1&rep=c",
           "port": None, "first": None, "final": None, "waited_s": 0.0,
           "timed_out_wait": False, "ts": time.time(),
           "since_pids": set(chrome_snapshot())}
    t0 = time.time()
    dbg = os.path.join(sdir, "collect_debug.log")
    if os.path.exists(dbg):
        os.remove(dbg)
    th = threading.Thread(target=collect,
                          args=(task, seed, t0, out, dbg), daemon=True)
    th.start()
    proc = subprocess.Popen(
        [OMP_BIN, "-p", prompt, "--model", MODEL, "--session-dir", sdir],
        cwd=OMP_CWD, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        rc = proc.wait(timeout=OMP_CAP_S)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        rc = 124
    wall = time.time() - t0
    out["cdp_stop"] = time.time() + SETTLE_S + 8
    th.join(timeout=SETTLE_S + 45)
    fin = out.get("final")

    cdp_row = {k: v for k, v in out.items()
               if k not in ("since_pids", "cdp_stop")}
    with open(CDP_MAIN, "a") as f:
        f.write(json.dumps(cdp_row) + "\n")

    fq = (fin or out.get("first") or {}).get("query")
    row = {"task": task, "seed": seed, "arm": ARM, "inst": len(done_tags),
           "tag": tag, "url": url, "omp_exit": rc, "wall_s": round(wall, 1),
           "done": (fin or {}).get("done"),
           "raw": (fin or {}).get("raw"),
           "fin_reward": (fin or {}).get("fin"),
           "reason": (fin or {}).get("reason"),
           "query": fq,
           "cdp_err": None if fin else "no final page state",
           "session": sdir, "ts": time.time()}
    with open(MAIN, "a") as f:
        f.write(json.dumps(row) + "\n")
    return row


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "plan"
    if mode == "plan":
        for t, seeds in seed_plan():
            print(t, seeds)
        return
    if mode == "sanity":
        seed = 3306495589  # throwaway, /dev/urandom-drawn, outside the frozen plan
        os.makedirs(SANITY, exist_ok=True)
        global MAIN, CDP_MAIN
        MAIN = os.path.join(SANITY, "main_v4sanity.jsonl")
        CDP_MAIN = os.path.join(SANITY, "main_v4sanity.jsonl.cdp")
        row = run_session("click-button", seed, SANITY)
        print(json.dumps(row, indent=1))
        return
    if mode == "sweep":
        n = 0
        for t, seeds in seed_plan():
            for s in seeds:
                r = run_session(t, s, RUNS)
                if r:
                    n += 1
                    raw = r["raw"]
                    print(f"[{n:3d}] {r['tag']} exit={r['omp_exit']} "
                          f"wall={r['wall_s']}s raw={raw} "
                          f"reason={r['reason']} q={str(r['query'])[:50]!r}",
                          flush=True)
        print("SWEEP DONE", n)
        return


if __name__ == "__main__":
    main()

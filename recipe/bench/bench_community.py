#!/usr/bin/env python3
"""community-v1 (tensorfold.dev/benchmarks, docs/community-benchmarks.md) re-implemented against an OpenAI-compatible server.
Suite (zig/src/server/benchmark_suite.zig): fixtures code (completion) + chat (chat), max_tokens 256, 1 warm-up + 5 repeats per cell,
temperatures 1.0 and 0.0, seed_base 1234, top_k 20, top_p 0.95 when sampled, thinking off, concurrency 1.
Rate = (completion_tokens-1)/(last nonempty delivery - first nonempty delivery); cell value = median of the 5 repeats.
usage: bench_community.py [--url http://127.0.0.1:18090] [--model ID] [--temps 1,0] [--reps 5] [--out FILE]"""
import argparse, http.client, json, statistics, time, urllib.parse
ap = argparse.ArgumentParser()
ap.add_argument("--url", default="http://127.0.0.1:18090"); ap.add_argument("--model", default="GLM-5.3-EXL3-3.0bpw")
ap.add_argument("--temps", default="1,0"); ap.add_argument("--reps", type=int, default=5); ap.add_argument("--tokens", type=int, default=256)
ap.add_argument("--out", default=""); a = ap.parse_args()
FIX = [("code", False, "Write a short Python function that computes the Fibonacci sequence and explain it."),
       ("chat", True, "Explain how matrix multiplication uses a GPU in plain English, then give a small numerical example.")]
u = urllib.parse.urlparse(a.url)


def attempt(chat, prompt, temp, seed):
    body = {"model": a.model, "max_tokens": a.tokens, "temperature": temp, "seed": seed, "stream": True,
            "stream_options": {"include_usage": True}, "chat_template_kwargs": {"enable_thinking": False}}
    if temp > 0:
        body.update(top_k=20, top_p=0.95)
    if chat:
        body["messages"] = [{"role": "user", "content": prompt}]
        path = "/v1/chat/completions"
    else:
        body["prompt"] = prompt
        path = "/v1/completions"
    c = http.client.HTTPConnection(u.hostname, u.port, timeout=600)
    t0 = time.perf_counter()
    c.request("POST", path, json.dumps(body), {"content-type": "application/json"})
    r = c.getresponse()
    if r.status != 200:
        return {"status": "error", "error": "HTTP %d %s" % (r.status, r.read()[:300].decode(errors="replace"))}
    first = last = None
    usage = None
    fin = None
    while True:
        line = r.readline()
        if not line:
            break
        line = line.strip()
        if not line.startswith(b"data:"):
            continue
        d = line[5:].strip()
        if d == b"[DONE]":
            break
        j = json.loads(d)
        usage = j.get("usage") or usage
        for ch in j.get("choices") or []:
            if chat:
                dl = ch.get("delta") or {}
                txt = dl.get("content") or dl.get("reasoning_content")
            else:
                txt = ch.get("text")
            if txt:
                now = time.perf_counter()
                first = first or now
                last = now
            fin = ch.get("finish_reason") or fin
    end = time.perf_counter()
    if not usage or first is None or last is None or last <= first:
        return {"status": "error", "error": "no usage or unmeasurable delivery"}
    n = usage["completion_tokens"]
    return {"status": "ok" if n >= a.tokens else "early_eos", "tokens": n, "prompt_tokens": usage.get("prompt_tokens"),
            "tps": (n - 1) / (last - first), "ttft": first - t0, "e2e": end - t0, "finish": fin}


res = []
for temp in [float(x) for x in a.temps.split(",")]:
    for fid, chat, prompt in FIX:
        w = attempt(chat, prompt, temp, 1234)
        if w["status"] == "error":
            print("%-5s T=%g: warm-up failed: %s" % (fid, temp, w["error"]), flush=True)
            res.append({"fixture": fid, "temperature": temp, "warmup_error": w["error"]})
            continue
        rows = [attempt(chat, prompt, temp, 1234 + i) for i in range(a.reps)]
        ok = [r for r in rows if r["status"] == "ok"]
        complete = len(ok) == len(rows)
        med = lambda k: statistics.median(r[k] for r in ok) if complete else None
        cell = {"fixture": fid, "temperature": temp, "attempts": rows, "tps_median": med("tps"), "ttft_median": med("ttft"), "complete": complete}
        res.append(cell)
        tps = ", ".join("%.1f" % r["tps"] for r in rows if "tps" in r)
        mt = cell["tps_median"]
        print("%-5s T=%g: median %s tok/s, first text %s s | reps %s | tokens %s | %s" % (
            fid, temp, round(mt, 2) if mt else None, round(cell["ttft_median"], 3) if cell["ttft_median"] else None, tps,
            [r.get("tokens") for r in rows], "complete" if complete else "INCOMPLETE " + str([r["status"] for r in rows])), flush=True)
if a.out:
    json.dump(res, open(a.out, "w"), indent=1)

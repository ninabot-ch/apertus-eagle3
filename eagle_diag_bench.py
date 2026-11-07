"""EAGLE acceptance and throughput by prompt family, against an already-running vLLM server (stdlib only).

Families (each optional, at least one required):
  --generic  JSONL, one {"prompt": "<text>"} per line
  --devops   JSONL, one {"prompt": "<text>"} per line
  --agentic  JSONL, one {"prompt": [<chat messages>], "tools": [<tool schemas>]} per line
For each family and concurrency level: output tokens/s, and mean acceptance length read from the vLLM Prometheus
counters (spec_decode_num_drafts / num_accepted_tokens / ..._per_pos) before and after the run.
Greedy decoding (temperature 0). Run the same script against a server started without speculative decoding to get
the baseline throughput.

usage: eagle_diag_bench.py --url http://127.0.0.1:8000 --model MODEL --generic G.jsonl --out res.json
       [--devops D.jsonl] [--agentic A.jsonl] [--conc 1,8] [--n-agentic 96] [--max-tokens 512]
"""
import argparse, concurrent.futures as cf, json, random, re, time, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--url", required=True)
ap.add_argument("--model", required=True)
ap.add_argument("--generic")
ap.add_argument("--devops")
ap.add_argument("--agentic")
ap.add_argument("--out", required=True)
ap.add_argument("--conc", default="1,8")
ap.add_argument("--n-agentic", type=int, default=96)
ap.add_argument("--max-tokens", type=int, default=512)
a = ap.parse_args()
if not (a.generic or a.devops or a.agentic):
    ap.error("give at least one of --generic, --devops, --agentic")


def metrics():
    txt = urllib.request.urlopen(a.url + "/metrics", timeout=30).read().decode()
    m = {}
    for line in txt.splitlines():
        r = re.match(r'^vllm:spec_decode_num_(drafts|draft_tokens|accepted_tokens|accepted_tokens_per_pos)_total\{([^}]*)\} ([0-9.e+]+)', line)
        if r:
            pos = re.search(r'position="(\d+)"', r.group(2))
            m[r.group(1) + (f"_{pos.group(1)}" if pos else "")] = float(r.group(3))
    return m


def load():
    rnd = random.Random(0)
    fams = {}
    for name, path in (("generic", a.generic), ("devops", a.devops)):
        if path:
            fams[name] = [{"messages": [{"role": "user", "content": json.loads(l)["prompt"]}]}
                          for l in open(path)][:48]
    if a.agentic:
        rows = [json.loads(l) for l in open(a.agentic)]
        rnd.shuffle(rows)
        n = []
        for r in rows[: a.n_agentic]:
            req = {"messages": r["prompt"]}
            if r.get("tools"):
                req["tools"] = r["tools"]
            n.append(req)
        fams["agentic"] = n
    return fams


def one(req):
    body = dict(req, model=a.model, max_tokens=a.max_tokens, temperature=0)
    t = time.time()
    r = json.load(urllib.request.urlopen(urllib.request.Request(
        a.url + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=900))
    return r["usage"]["completion_tokens"], time.time() - t


res = {}
for fam, reqs in load().items():
    for c in [int(x) for x in a.conc.split(",")]:
        reqs_c = reqs[: max(16, c * 4)] if c > 1 else reqs[:16]
        m0, t0 = metrics(), time.time()
        errs = 0
        with cf.ThreadPoolExecutor(c) as ex:
            outs = []
            for f in [ex.submit(one, q) for q in reqs_c]:
                try:
                    outs.append(f.result())
                except Exception:
                    errs += 1
        dt, m1 = time.time() - t0, metrics()
        dm = {k: m1.get(k, 0) - m0.get(k, 0) for k in m1}
        tok = sum(o[0] for o in outs)
        e = {"n": len(reqs_c), "errors": errs, "out_tokens": tok, "wall_s": round(dt, 1),
             "out_tok_s": round(tok / dt, 1) if dt else None}
        if dm.get("drafts"):
            e["mean_accept_len"] = round(1 + dm["accepted_tokens"] / dm["drafts"], 3)
            e["accept_per_pos"] = [round(dm.get(f"accepted_tokens_per_pos_{i}", 0) / dm["drafts"], 3) for i in range(8)
                                   if f"accepted_tokens_per_pos_{i}" in dm]
        res[f"{fam}-c{c}"] = e
        print(fam, c, json.dumps(e), flush=True)
json.dump(res, open(a.out, "w"), indent=1)

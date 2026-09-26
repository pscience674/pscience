"""Sanity checks for a refreshed PreScience corpus split (all_papers.json [+ grit embeddings])."""
import sys, json, pickle, collections, statistics, random
import numpy as np

split_dir = sys.argv[1]
check_emb = len(sys.argv) > 2 and sys.argv[2] == "emb"

with open(f"{split_dir}/all_papers.json") as f:
    raw = json.load(f)
papers_list = raw["data"] if isinstance(raw, dict) and "data" in raw else raw
meta = raw.get("metadata") if isinstance(raw, dict) else None
print("metadata:", json.dumps(meta)[:500] if meta else None)
papers = {p["corpus_id"]: p for p in papers_list}
print(f"papers: {len(papers_list)}  unique corpus_ids: {len(papers)}")

targets = [p for p in papers_list if "target" in p.get("roles", [])]
print(f"targets: {len(targets)}")
print("example target keys:", sorted(targets[0].keys()))
print("example author keys:", sorted(targets[0]["authors"][0].keys()))
print("example key_ref keys:", sorted(targets[0]["key_references"][0].keys()) if targets[0].get("key_references") else None)

role_counts = collections.Counter(r for p in papers_list for r in p.get("roles", []))
print("role counts:", dict(role_counts))

issues = collections.defaultdict(list)

# Dates / split window
tdates = sorted(p["date"] for p in targets)
print(f"target date range: {tdates[0]} .. {tdates[-1]}")
month_counts = collections.Counter(d[:7] for d in tdates)
print("targets per month:", dict(sorted(month_counts.items())))

# Categories
cat_key = next((k for k in ("categories", "arxiv_categories", "primary_category") if k in targets[0]), None)
if cat_key:
    import os, re as _re
    cat_re = _re.compile(os.environ.get("SANITY_CATS", r"^(cs\.CL|cs\.LG|cs\.AI|cs\.CV|cs\.IR|cs\.NE)$"))
    cats = collections.Counter()
    for p in targets:
        c = p[cat_key] if isinstance(p[cat_key], list) else str(p[cat_key]).split()
        cats.update(c)
        if not any(cat_re.search(x) for x in c):
            issues["target without an in-domain category"].append(p["corpus_id"])
    print(f"top categories ({cat_key}):", cats.most_common(12))
else:
    print("no category field found")

# Text quality
titles = collections.Counter((p.get("title") or "").strip().lower() for p in targets)
for p in targets:
    if not (p.get("title") or "").strip():
        issues["target empty title"].append(p["corpus_id"])
    if len((p.get("abstract") or "").split()) < 30:
        issues["target abstract < 30 words"].append(p["corpus_id"])
dups = [t for t, c in titles.items() if c > 1 and t]
print(f"duplicate target titles: {len(dups)}  e.g. {dups[:3]}")
arx = collections.Counter(p.get("arxiv_id") for p in targets if p.get("arxiv_id"))
print(f"duplicate arxiv_ids among targets: {sum(1 for c in arx.values() if c > 1)}")
wc = [len((p.get("abstract") or "").split()) + len((p.get("title") or "").split()) for p in targets]
print(f"avg words (title+abstract): {statistics.mean(wc):.1f}")

# Key references
nk = [len(p.get("key_references") or []) for p in targets]
print(f"key refs per target: avg {statistics.mean(nk):.2f}, median {statistics.median(nk)}, min {min(nk)}, max {max(nk)}")
for p in targets:
    for r in p.get("key_references") or []:
        q = papers.get(r["corpus_id"])
        if q is None:
            issues["key_ref missing from corpus"].append((p["corpus_id"], r["corpus_id"]))
        elif q["date"] >= p["date"]:
            issues["LEAK key_ref date >= target date"].append((p["corpus_id"], r["corpus_id"], q["date"], p["date"]))
        if "num_citations" not in r:
            issues["key_ref missing num_citations"].append((p["corpus_id"], r["corpus_id"]))

# Authors
na = [len(p.get("authors") or []) for p in targets]
print(f"authors per target: avg {statistics.mean(na):.2f}, max {max(na)}")
uniq_auth = set()
hist_lens = []
for p in targets:
    if not p.get("authors"):
        issues["target no authors"].append(p["corpus_id"])
    for a in p.get("authors") or []:
        uniq_auth.add(a["author_id"])
        ph = a.get("publication_history", [])
        hist_lens.append(len(ph))
        for pid in ph:
            q = papers.get(pid)
            if q is None:
                issues["pub_history paper missing"].append((p["corpus_id"], pid))
            elif q["date"] >= p["date"]:
                issues["LEAK pub_history date >= target date"].append((p["corpus_id"], pid, q["date"], p["date"]))
        if not all(k in a for k in ("h_index", "num_papers", "num_citations")):
            issues["author missing bibliometrics"].append((p["corpus_id"], a["author_id"]))
        elif a["num_papers"] < len(ph):
            issues["author num_papers < len(pub_history) (bibliometrics not time-aligned?)"].append((p["corpus_id"], a["author_id"], a["num_papers"], len(ph)))
print(f"unique target authors: {len(uniq_auth)}; author history avg {statistics.mean(hist_lens):.1f}, median {statistics.median(hist_lens)}, zero-history frac {sum(h == 0 for h in hist_lens) / len(hist_lens):.3f}")

# Citation trajectories
traj_lens = collections.Counter()
c12 = []
for p in targets:
    t = p.get("citation_trajectory")
    if t is None:
        issues["target missing citation_trajectory"].append(p["corpus_id"])
        continue
    vals = [x if isinstance(x, (int, float)) else x.get("count", x.get("citations", 0)) for x in t] if isinstance(t, list) else list(t.values())
    traj_lens[len(vals)] += 1
    if any(b < a for a, b in zip(vals, vals[1:])):
        issues["citation_trajectory not monotone"].append(p["corpus_id"])
    if len(vals) >= 12:
        c12.append(vals[11])
print("citation_trajectory lengths:", dict(sorted(traj_lens.items())))
print("example trajectory:", targets[0].get("citation_trajectory"))
if c12:
    print(f"12-month citations (n={len(c12)}): mean {statistics.mean(c12):.2f}, median {statistics.median(c12)}, max {max(c12)}")

# Topics
tk = next((k for k in ("topics", "topic_labels", "topic") if k in targets[0]), None)
if tk:
    nt = [len(p.get(tk) or []) for p in targets]
    allt = collections.Counter(t for p in targets for t in (p.get(tk) or []))
    print(f"topics ({tk}): avg per target {statistics.mean(nt):.2f}, zero-topic frac {sum(n == 0 for n in nt) / len(nt):.3f}, distinct {len(allt)}")
    print("  top topics:", allt.most_common(8))
else:
    print("no topics field found")

# Orphans
reach = set()
for p in papers_list:
    if "target" in p.get("roles", []):
        reach.add(p["corpus_id"])
    for r in p.get("key_references") or []:
        reach.add(r["corpus_id"])
    for a in p.get("authors") or []:
        reach.update(a.get("publication_history", []))
orph = [c for c in papers if c not in reach]
print(f"orphan papers: {len(orph)}")

# Random spot-check sample
random.seed(0)
for p in random.sample(targets, 2):
    print("\n--- sample target", p["corpus_id"], p["date"], p.get("arxiv_id"))
    print("  title:", p.get("title"))
    print("  authors:", [(a.get("name"), a["author_id"], len(a.get("publication_history", [])), a.get("h_index")) for a in p["authors"]][:6])
    for r in p["key_references"][:4]:
        q = papers.get(r["corpus_id"], {})
        print("  keyref:", q.get("date"), (q.get("title") or "")[:90])

if check_emb:
    with open(f"{split_dir}/all_papers.grit_embeddings.pkl", "rb") as f:
        emb = pickle.load(f)
    emb = emb["data"] if isinstance(emb, dict) and "data" in emb else emb
    emb = {k: (v["key"].reshape(-1) if isinstance(v, dict) else v) for k, v in emb.items()}
    print(f"\nembeddings: type {type(emb).__name__}, n={len(emb)}")
    keys = list(emb.keys()) if isinstance(emb, dict) else None
    v0 = np.asarray(emb[keys[0]]) if keys else None
    print(f"  dim {v0.shape}, dtype {v0.dtype}")
    missing = [c for c in papers if c not in emb]
    print(f"  papers missing embeddings: {len(missing)} ({len(missing) / len(papers):.4f})")
    samp = random.sample(keys, min(20000, len(keys)))
    M = np.stack([np.asarray(emb[k], dtype=np.float32) for k in samp])
    norms = np.linalg.norm(M, axis=1)
    print(f"  NaN rows: {np.isnan(M).any(1).sum()}, zero rows: {(norms == 0).sum()}, norm mean {norms.mean():.3f} std {norms.std():.3f}")
    Mn = M / np.maximum(norms[:, None], 1e-9)
    dupe = (np.abs(Mn[:2000] @ Mn[:2000].T - 1) < 1e-6).sum() - 2000
    print(f"  identical-embedding pairs among 2000 sampled: {dupe // 2}")
    # Semantic check: a target's nearest key reference should be closer than a random paper
    closer, tot = 0, 0
    for p in random.sample(targets, 500):
        if p["corpus_id"] not in emb:
            continue
        e = np.asarray(emb[p["corpus_id"]], dtype=np.float32); e /= np.linalg.norm(e)
        refs = [np.asarray(emb[r["corpus_id"]], dtype=np.float32) for r in p["key_references"] if r["corpus_id"] in emb]
        if not refs:
            continue
        rs = max(float(e @ (r / np.linalg.norm(r))) for r in refs)
        rnd = Mn[random.randrange(len(Mn))]
        closer += rs > float(e @ rnd); tot += 1
    print(f"  target closer to its key ref than to a random paper: {closer}/{tot}")

print("\n=== ISSUES ===")
if not issues:
    print("none")
for k, v in issues.items():
    print(f"  {k}: {len(v)}  e.g. {v[:3]}")

"""Stage-by-stage sanity checks for an in-progress corpus build (all_papers.stageNN.json).

Prints counts, date ranges, category coverage, and the invariants that should already hold at that stage.
"""
import re
import argparse
import statistics
from collections import Counter

import utils


def main():
    parser = argparse.ArgumentParser("Validate an intermediate corpus stage file.")
    parser.add_argument("--input_path", type=str, required=True, help="Path to all_papers.stageNN.json")
    parser.add_argument("--start_date", type=str, required=True, help="Target window start (inclusive)")
    parser.add_argument("--end_date", type=str, required=True, help="Target window end (exclusive)")
    parser.add_argument("--category_regex", type=str, required=True, help="Regex a target's categories must match")
    args = parser.parse_args()

    stage = int(re.search(r"stage(\d+)", args.input_path).group(1))
    papers, metadata = utils.load_json(args.input_path)
    by_id = {p["corpus_id"]: p for p in papers}
    targets = [p for p in papers if "target" in p.get("roles", [])]
    problems = Counter()
    print(f"stage {stage}: {len(papers)} papers ({len(by_id)} unique ids), {len(targets)} targets; last script: {metadata[-1]['script'].split('/')[-1] if metadata else None}")
    print("roles:", dict(Counter(r for p in papers for r in p.get("roles", []))))

    dates = sorted(p["date"] for p in targets)
    print(f"target dates {dates[0]} .. {dates[-1]}; per month: {dict(sorted(Counter(d[:7] for d in dates).items()))}")
    cat_re = re.compile(args.category_regex)
    for p in targets:
        if not (args.start_date <= p["date"] < args.end_date):
            problems["target outside date window"] += 1
        if not any(cat_re.search(c) for c in p.get("categories", [])):
            problems["target without a matching category"] += 1
        if not (p.get("title") or "").strip() or not (p.get("abstract") or "").strip():
            problems["target missing title/abstract"] += 1
    print("top categories:", Counter(c for p in targets for c in p.get("categories", [])).most_common(10))
    if len(by_id) != len(papers):
        problems["duplicate corpus ids"] += len(papers) - len(by_id)

    if stage >= 2:
        nk = [len(p.get("key_references") or []) for p in targets]
        print(f"key refs/target: avg {statistics.mean(nk):.2f}, median {statistics.median(nk)}, min {min(nk)}, max {max(nk)}")
        for p in targets:
            refs = p.get("key_references") or []
            if not 1 <= len(refs) <= 10:
                problems["target key_refs outside [1,10]"] += 1
            for r in refs:
                q = by_id.get(r["corpus_id"])
                if q is None:
                    problems["key ref missing from corpus"] += 1
                elif q["date"] >= p["date"]:
                    problems["key ref dated on/after target"] += 1

    if stage >= 3:
        na = [len(p.get("authors") or []) for p in targets]
        hist = [len(a.get("publication_history", [])) for p in targets for a in p.get("authors") or []]
        print(f"authors/target: avg {statistics.mean(na):.2f}, max {max(na)}; history/author avg {statistics.mean(hist):.1f}, median {statistics.median(hist)}, zero-history frac {sum(h == 0 for h in hist) / len(hist):.3f}")
        for p in targets:
            if not p.get("authors"):
                problems["target without authors"] += 1
            for a in p.get("authors") or []:
                for pid in a.get("publication_history", []):
                    q = by_id.get(pid)
                    if q is None:
                        problems["pub history paper missing"] += 1
                    elif q["date"] >= p["date"]:
                        problems["pub history dated on/after target"] += 1

    print("PROBLEMS:", dict(problems) if problems else "none")


if __name__ == "__main__":
    main()

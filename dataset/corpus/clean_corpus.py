"""Final corpus cleanup: flatten list-encoded middle names and drop temporally invalid key references.

Author names from the S2 pipeline sometimes embed middle names as a JSON list (e.g. 'Hoang ["M."] Ngo');
these are flattened to 'Hoang M. Ngo'. Key references dated on or after the citing paper (typically a
journal-version date overriding the preprint date) are dropped, and papers orphaned by the drop are removed.
"""
import os
import re
import json
import argparse

import utils

BRACKET_RE = re.compile(r"\[[^\]]*\]")


def flatten_name(name):
    def repl(match):
        segment = match.group(0)
        try:
            parts = json.loads(segment)
        except json.JSONDecodeError:
            parts = [segment.strip("[]").replace("'", "").replace('"', "")]
        return " ".join(str(p) for p in parts if p)
    return re.sub(r"\s+", " ", BRACKET_RE.sub(repl, name)).strip()


def main():
    parser = argparse.ArgumentParser("Clean author names and drop temporally invalid key references.")
    parser.add_argument("--input_path", type=str, required=True, help="Path to all_papers.json to clean")
    parser.add_argument("--output_path", type=str, required=True, help="Path to write the cleaned all_papers.json")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output if it exists")
    parser.add_argument("--sd2publications_path", type=str, default=None, help="If given, prune author publication lists to kept papers and rewrite this file in place")
    args = parser.parse_args()

    all_papers, metadata = utils.load_json(args.input_path)
    papers = {p["corpus_id"]: p for p in all_papers}

    names_fixed = 0
    for p in all_papers:
        for author in p.get("authors") or []:
            name = author.get("name")
            if name and "[" in name:
                author["name"] = flatten_name(name)
                names_fixed += 1
    utils.log(f"Flattened {names_fixed} author names")

    dropped = []
    for p in all_papers:
        refs = p.get("key_references")
        if not refs:
            continue
        kept = [r for r in refs if r["corpus_id"] in papers and papers[r["corpus_id"]]["date"] < p["date"]]
        if len(kept) != len(refs):
            dropped.extend((p["corpus_id"], r["corpus_id"]) for r in refs if r not in kept)
            p["key_references"] = kept
    utils.log(f"Dropped {len(dropped)} key references dated on/after the citing paper: {dropped[:10]}")

    emptied = [p["corpus_id"] for p in all_papers if "target" in p["roles"] and not p.get("key_references")]
    if emptied:
        utils.log(f"Removing {len(emptied)} targets left with no key references")
    emptied = set(emptied)

    # Remove orphans iteratively: dropping refs/targets can orphan companion papers.
    kept_papers = [p for p in all_papers if p["corpus_id"] not in emptied]
    while True:
        reachable = set()
        for p in kept_papers:
            if "target" in p["roles"]:
                reachable.add(p["corpus_id"])
            for ref in p.get("key_references") or []:
                reachable.add(ref["corpus_id"])
            for author in p.get("authors") or []:
                reachable.update(author.get("publication_history", []))
        pruned = [p for p in kept_papers if p["corpus_id"] in reachable]
        if len(pruned) == len(kept_papers):
            break
        kept_papers = pruned
    utils.log(f"Papers: {len(all_papers)} -> {len(kept_papers)}")

    metadata = (metadata or []) + [{"script": os.path.abspath(__file__), "args": vars(args)}]
    utils.save_json(kept_papers, args.output_path, metadata=metadata, overwrite=args.overwrite)
    if args.sd2publications_path:
        sd2publications, sd_meta = utils.load_json(args.sd2publications_path)
        kept_ids = {p["corpus_id"] for p in kept_papers}
        removed = 0
        for author_id, pubs in sd2publications.items():
            if pubs:
                before = len(pubs)
                sd2publications[author_id] = [c for c in pubs if c in kept_ids]
                removed += before - len(sd2publications[author_id])
        utils.log(f"Pruned {removed} publication entries pointing at removed papers from {args.sd2publications_path}")
        utils.save_json(sd2publications, args.sd2publications_path, metadata=sd_meta, overwrite=True)


if __name__ == "__main__":
    main()

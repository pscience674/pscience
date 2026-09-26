"""Probe empty-label papers with a reason-included prompt to diagnose taxonomy gaps.

Reads a Phase 1 JSONL output (records with `topics: []` indicate gpt-5.4 refused to
label), samples a subset, re-queries gpt-5.4 with a schema that includes a `reason`
field, and saves `{corpus_id, topics, reason}` per paper. The reasons are then used
to identify recurring taxonomy gaps and propose new topics.

Resumable: appends to --output_path and skips corpus_ids already present.
"""

import argparse
import json
import os
import random
import time
import concurrent.futures as cf

from tqdm import tqdm
import openai

import utils
from dataset.corpus.assign_topics import load_topics, strip_code_fences, safe_text, with_retries


def build_probe_system_prompt(topics):
    bullets = "\n".join(f"- {t}" for t in topics)
    return ("You are a scientific paper topic classifier. Given a paper title and abstract, assign all topics from the list below that clearly apply.\n\nTopics:\n" + bullets + "\n\nRules:\n"
            "1) Output ONLY compact JSON of the form {\"topics\": [...], \"reason\": \"<1-2 sentences>\"}.\n"
            "2) Each string in the list MUST exactly match one topic from the list above (same spelling and punctuation).\n"
            "3) If no topic clearly applies, output {\"topics\": [], \"reason\": \"<brief reason no topic applies>\"}.\n"
            "4) Never invent topics outside the list.\n"
            "5) List all applicable topics in descending order of relevance. Papers commonly span multiple topics — a method, a task, a domain, and a data concern can each map to different topics. Include every topic the paper directly addresses, not just the primary focus. Exclude topics that are only mentioned in passing or tangentially adjacent.\n"
            "6) The \"reason\" field must briefly explain your reasoning. For empty results, explain why no topic fits.\n")


def parse_probe_response(raw, topic_set):
    cleaned = strip_code_fences(raw)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        return {"topics": [], "reason": f"PARSE_ERROR: {raw[:200]}"}
    raw_topics = data.get("topics", []) if isinstance(data, dict) else []
    seen, topics = set(), []
    if isinstance(raw_topics, list):
        for t in raw_topics:
            if isinstance(t, str) and t in topic_set and t not in seen:
                seen.add(t)
                topics.append(t)
    reason = str(data.get("reason", "")).strip() if isinstance(data, dict) else ""
    return {"topics": topics, "reason": reason}


def query_probe(client, model, system_prompt, title, abstract, topic_set):
    user_content = f"Title: {safe_text(title, 300)}\nAbstract: {safe_text(abstract, 2000)}\n\nList all applicable topics in descending order of relevance."
    response = client.chat.completions.create(model=model, messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_content}])
    return parse_probe_response(response.choices[0].message.content.strip(), topic_set)


def load_existing(output_path):
    if not os.path.exists(output_path):
        return set()
    done = set()
    with open(output_path) as f:
        for line in f:
            try:
                done.add(json.loads(line)["corpus_id"])
            except (json.JSONDecodeError, KeyError):
                continue
    return done


def load_empty_corpus_ids(input_path):
    empties = []
    with open(input_path) as f:
        for line in f:
            r = json.loads(line)
            if not r["topics"]:
                empties.append(r["corpus_id"])
    return empties


def main():
    parser = argparse.ArgumentParser(description="Probe empty-label papers with a reason-included prompt to identify taxonomy gaps.")
    parser.add_argument("--input_path", type=str, required=True, help="Phase 1 JSONL (records with topics==[] will be probed)")
    parser.add_argument("--output_path", type=str, required=True, help="JSONL output with {corpus_id, topics, reason}")
    parser.add_argument("--topics_path", type=str, default="dataset/corpus/topics_list.txt")
    parser.add_argument("--model", type=str, default="gpt-5.4-2026-03-05")
    parser.add_argument("--num_probe", type=int, default=1000, help="Number of empty-label papers to sample")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_workers", type=int, default=64)
    parser.add_argument("--data_dir", type=str, default="data/prescience-ai", help="Dataset directory (with train/ and test/ subdirectories)")
    args = parser.parse_args()

    topics = load_topics(args.topics_path)
    topic_set = set(topics)
    utils.log(f"Loaded {len(topics)} topics")

    empties = load_empty_corpus_ids(args.input_path)
    utils.log(f"Found {len(empties)} empty-label corpus_ids in {args.input_path}")
    rng = random.Random(args.seed)
    rng.shuffle(empties)
    sample_ids = set(empties[:args.num_probe])
    utils.log(f"Sampled {len(sample_ids)} empties for probing")

    already = load_existing(args.output_path)
    to_probe_ids = sample_ids - already
    utils.log(f"Already probed: {len(already & sample_ids)}; to probe: {len(to_probe_ids)}")

    utils.log("Loading corpus for title/abstract lookup")
    by_id = {}
    for split in ["train", "test"]:
        papers, _, _ = utils.load_corpus(data_dir=args.data_dir, split=split, embedding_type=None, load_sd2publications=False)
        for p in papers:
            if p["corpus_id"] in to_probe_ids and p["corpus_id"] not in by_id:
                by_id[p["corpus_id"]] = p
    utils.log(f"Resolved {len(by_id)} / {len(to_probe_ids)} papers")

    client = openai.OpenAI()
    system_prompt = build_probe_system_prompt(topics)
    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    out_f = open(args.output_path, "a")
    written = 0

    def work(cid):
        p = by_id.get(cid)
        if p is None:
            return {"corpus_id": cid, "topics": [], "reason": "PAPER_NOT_FOUND"}
        result = with_retries(lambda: query_probe(client, args.model, system_prompt, p.get("title", ""), p.get("abstract", ""), topic_set))
        if result is None:
            return {"corpus_id": cid, "topics": [], "reason": "REQUEST_FAILED"}
        return {"corpus_id": cid, **result}

    try:
        with cf.ThreadPoolExecutor(max_workers=args.max_workers) as ex:
            futures = [ex.submit(work, cid) for cid in to_probe_ids]
            for future in tqdm(cf.as_completed(futures), total=len(futures), desc=f"Probing ({args.model})"):
                rec = future.result()
                out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                out_f.flush()
                written += 1
    finally:
        out_f.close()
    utils.log(f"Wrote {written} probe records to {args.output_path}")


if __name__ == "__main__":
    main()

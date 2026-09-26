"""RLM baseline for prior work prediction. Loads the date-filtered candidate pool into a Python REPL and lets a Recursive Language Model rank likely key references."""

import os
import re
import ast
import json
import atexit
import pickle
import random
import tempfile
import argparse
import traceback
import threading
import concurrent.futures as cf

from tqdm import tqdm
from rlm import RLM

import utils
from task_priorwork_prediction.dataset import create_evaluation_instances, get_preexisting_publications_for_author

random.seed(42)

FINAL_RE = re.compile(r"FINAL[:_]?(?:VAR\([^)]*\))?\s*(\[[^\]]*\])", re.DOTALL)

PICKLE_PATHS = []
PICKLE_PATHS_LOCK = threading.Lock()
SAVE_LOCK = threading.Lock()


def _cleanup_pickles():
    for path in PICKLE_PATHS:
        if os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                pass


atexit.register(_cleanup_pickles)


def build_candidate_pool(target_corpus_id, target_date, all_papers):
    """Return the per-instance candidate pool: every paper strictly before target_date, excluding the target itself. Non-target papers may lack authors/categories — use safe defaults. Each entry now also exposes `author_ids` so the LLM can join with `pubs_by_author`."""
    return [{
        "corpus_id": p["corpus_id"],
        "title": p.get("title") or "",
        "abstract": p.get("abstract") or "",
        "date": p["date"],
        "author_ids": [a.get("author_id") for a in (p.get("authors") or []) if isinstance(a, dict) and a.get("author_id")],
        "author_names": [a.get("name") for a in (p.get("authors") or []) if isinstance(a, dict) and a.get("name")],
        "categories": p.get("categories") or [],
    } for p in all_papers if p["date"] < target_date and p["corpus_id"] != target_corpus_id]


def build_pubs_by_author(candidate_pool):
    """Index author_id -> list of corpus_ids (every author's complete pre-cutoff publication history within `candidate_pool`)."""
    pubs = {}
    for paper in candidate_pool:
        for aid in paper["author_ids"]:
            pubs.setdefault(aid, []).append(paper["corpus_id"])
    return pubs


def build_author_history(target_authors, cutoff_date, sd2publications, all_papers_dict):
    """For each target author, collect their past publications (causally-available info only) with title/abstract/date/key-reference-corpus_ids. Returns dict of author_id -> list of paper records."""
    history = {}
    for aid in target_authors:
        past = get_preexisting_publications_for_author(aid, cutoff_date, sd2publications, all_papers_dict)
        pubs = []
        for cid in past:
            p = all_papers_dict[cid]
            pubs.append({"corpus_id": cid, "title": p.get("title") or "", "abstract": p.get("abstract") or "", "date": p["date"], "categories": p.get("categories") or [], "key_reference_corpus_ids": [r["corpus_id"] for r in (p.get("key_references") or [])]})
        history[aid] = pubs
    return history


def render_prompt(template, target_authors, target_date, top_n):
    """Fill the prompt template with ONLY causally-available info: author IDs and date. NO target title/abstract — those are written after the references are chosen and would leak."""
    return template.format(top_n=top_n, target_date=target_date, target_authors=", ".join(target_authors), n_authors=len(target_authors))


def _try_parse_list(text):
    """Try to parse text as a Python list literal or JSON list. Returns list or None."""
    text = text.strip()
    for parser in (ast.literal_eval, json.loads):
        try:
            obj = parser(text)
            if isinstance(obj, list):
                return obj
        except (ValueError, SyntaxError, json.JSONDecodeError):
            continue
    return None


def parse_final_answer(response_text, valid_corpus_ids):
    """Extract the predicted corpus_id list from the RLM response. The lib's `result.response` after `FINAL_VAR(predictions)` is `str(predictions)` (Python list repr); after `FINAL([...])` it's the bracketed text. Falls back to regex on a FINAL marker. Returns (predicted_ids, num_hallucinated)."""
    parsed = _try_parse_list(response_text)
    if parsed is None:
        match = FINAL_RE.search(response_text)
        if match is not None:
            parsed = _try_parse_list(match.group(1))
    if parsed is None:
        return [], 0
    valid_set = set(valid_corpus_ids)
    kept = [cid for cid in parsed if isinstance(cid, str) and cid in valid_set]
    hallucinated = len(parsed) - len(kept)
    return kept, hallucinated


def pad_predictions(predicted_ids, k, all_papers_dict, cutoff_date, exclude):
    """Pad ranked predictions to length k with random prior papers, mirroring baseline_frequency.py."""
    if len(predicted_ids) >= k:
        return predicted_ids[:k]
    excluded = set(predicted_ids) | set(exclude)
    available = [cid for cid in all_papers_dict if cid not in excluded and all_papers_dict[cid]["date"] < cutoff_date]
    needed = k - len(predicted_ids)
    extras = random.sample(available, min(needed, len(available)))
    return predicted_ids + extras


def write_pickle(candidate_pool):
    """Write candidate_pool to a tempfile, register for atexit cleanup, return path."""
    fd, path = tempfile.mkstemp(prefix="rlm_priorwork_", suffix=".pkl")
    os.close(fd)
    with open(path, "wb") as f:
        pickle.dump(candidate_pool, f)
    with PICKLE_PATHS_LOCK:
        PICKLE_PATHS.append(path)
    return path


def run_one_instance(job):
    """Process one evaluation instance: pool + author_history → RLM → ranked corpus_ids."""
    instance, args = job["instance"], job["args"]
    cutoff_date = instance["date"]
    target_authors = instance["author_ids"]
    candidate_pool = build_candidate_pool(instance["corpus_id"], cutoff_date, job["all_papers"])
    valid_ids = [c["corpus_id"] for c in candidate_pool]
    author_history = build_author_history(target_authors, cutoff_date, job["sd2publications"], job["all_papers_dict"])
    pubs_by_author = build_pubs_by_author(candidate_pool)

    pickle_path = write_pickle({"candidates": candidate_pool, "author_history": author_history, "pubs_by_author": pubs_by_author, "target_authors": target_authors, "target_date": cutoff_date})
    response_text = ""
    execution_time = None
    usage_summary = None

    try:
        setup_code = f"import pickle\n_payload = pickle.load(open({pickle_path!r}, 'rb'))\ncandidates = _payload['candidates']\ncorpus_by_id = {{c['corpus_id']: c for c in candidates}}\nauthor_history = _payload['author_history']\npubs_by_author = _payload['pubs_by_author']\ntarget_authors = _payload['target_authors']\ntarget_date = _payload['target_date']"
        env_kwargs = {"setup_code": setup_code}
        if args.environment == "ipython":
            env_kwargs["kernel_mode"] = args.ipython_kernel_mode
        rlm_kwargs = {"backend": args.backend, "backend_kwargs": {"model_name": args.model}, "environment": args.environment, "environment_kwargs": env_kwargs, "max_iterations": args.max_iterations, "max_depth": args.max_depth, "max_timeout": float(args.per_instance_timeout), "verbose": args.verbose}
        rlm = RLM(**rlm_kwargs)
        prompt = render_prompt(job["prompt_template"], target_authors, cutoff_date, args.top_n)
        result = rlm.completion(prompt)
        response_text = getattr(result, "response", "") or ""
        execution_time = getattr(result, "execution_time", None)
        usage_summary = getattr(result, "usage_summary", None)
        predicted_ids, hallucinated = parse_final_answer(response_text, valid_ids)
    except Exception as e:
        utils.log(f"Error on {instance['corpus_id']}: {type(e).__name__}: {e}")
        utils.log(traceback.format_exc())
        predicted_ids, hallucinated = [], 0

    num_model_predictions = len(predicted_ids)
    predicted_ids = pad_predictions(predicted_ids, args.k, job["all_papers_dict"], cutoff_date, [instance["corpus_id"]])
    predicted_scores = [1.0 / (i + 1) for i in range(len(predicted_ids))]

    return {"corpus_id": instance["corpus_id"], "gt_reference_ids": instance["gt_reference_ids"], "predicted_reference_ids": predicted_ids, "predicted_reference_scores": predicted_scores, "num_model_predictions": num_model_predictions, "num_hallucinated_ids": hallucinated, "candidate_pool_size": len(candidate_pool), "execution_time_seconds": execution_time, "usage_summary": str(usage_summary) if usage_summary is not None else None, "raw_response_truncated": response_text[-2000:]}


def main():
    parser = argparse.ArgumentParser(description="RLM baseline for prior work prediction")
    parser.add_argument("--data_dir", type=str, default="data/prescience-ai", help="Dataset directory (with train/ and test/ subdirectories)")
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"], help="Dataset split to use")
    parser.add_argument("--output_dir", type=str, default="data/task_priorwork_prediction/test/predictions", help="Where to save predictions JSON")
    parser.add_argument("--model", type=str, default="gpt-5-mini", help="Model name passed to RLM backend_kwargs (RLM paper/lib defaults: gpt-5, gpt-5-mini, gpt-5-nano)")
    parser.add_argument("--backend", type=str, default="openai", help="RLM backend (openai, anthropic, openrouter, ...)")
    parser.add_argument("--environment", type=str, default="ipython", choices=["local", "ipython"], help="RLM REPL environment. ipython+subprocess is thread-safe; local has an os.chdir race")
    parser.add_argument("--ipython_kernel_mode", type=str, default="subprocess", choices=["in_process", "subprocess"], help="ipython kernel mode. subprocess is required for thread isolation")
    parser.add_argument("--max_iterations", type=int, default=30, help="Max REPL turns before forcing final answer")
    parser.add_argument("--max_depth", type=int, default=1, help="Max RLM recursion depth")
    parser.add_argument("--top_n", type=int, default=100, help="Max number of corpus_ids the LM should return")
    parser.add_argument("--k", type=int, default=1000, help="Pad/truncate predictions to this length")
    parser.add_argument("--num_workers", type=int, default=4, help="Concurrent instances. Safe with --environment ipython --ipython_kernel_mode subprocess; use 1 for local environment (chdir race).")
    parser.add_argument("--per_instance_timeout", type=int, default=600, help="Seconds per RLM call (passed as max_timeout to RLM)")
    parser.add_argument("--max_instances", type=int, default=None, help="Cap on instances (use 10 for smoke)")
    parser.add_argument("--verbose", action="store_true", help="Verbose RLM trajectories")
    parser.add_argument("--prompt_template", type=str, default="task_priorwork_prediction/templates/rlm_priorwork_v3_causal.prompt", help="Path to the system prompt template file. v1/v2 prompts use target title+abstract — those leak the target's content, which is causally unavailable. v3_causal uses only authors+date, like the existing baselines.")
    parser.add_argument("--output_suffix", type=str, default=".causal", help="Suffix appended to predictions filename. Default '.causal' marks output as the causally-correct (no future-leakage) variant.")
    args = parser.parse_args()

    abs_output_dir = os.path.abspath(args.output_dir)
    os.makedirs(abs_output_dir, exist_ok=True)
    output_path = os.path.join(abs_output_dir, f"predictions.rlm.{args.model}{args.output_suffix}.json")
    utils.log(f"Output path resolved to absolute: {output_path}")

    utils.log(f"Loading corpus from {args.data_dir} (split={args.split})")
    all_papers, sd2publications, _ = utils.load_corpus(data_dir=args.data_dir, split=args.split, embedding_type=None, load_sd2publications=True)
    all_papers_dict = {p["corpus_id"]: p for p in all_papers}
    utils.log(f"Loaded {len(all_papers)} papers and {len(sd2publications)} author publication histories")

    base_cwd = os.getcwd()

    utils.log("Creating evaluation instances")
    evaluation_instances = create_evaluation_instances(all_papers, sd2publications, all_papers_dict)
    utils.log(f"Created {len(evaluation_instances)} evaluation instances")

    if args.max_instances is not None and args.max_instances < len(evaluation_instances):
        selected_indices = sorted(random.sample(range(len(evaluation_instances)), args.max_instances))
        utils.log(f"Randomly selected {len(selected_indices)} instances to evaluate")
    else:
        selected_indices = list(range(len(evaluation_instances)))

    with open(args.prompt_template, "r") as f:
        prompt_template = f.read()
    utils.log(f"Using prompt template: {args.prompt_template}")

    jobs = []
    for idx in selected_indices:
        date, instance = evaluation_instances[idx]
        instance = {**instance, "date": date}
        jobs.append({"instance": instance, "all_papers": all_papers, "all_papers_dict": all_papers_dict, "sd2publications": sd2publications, "prompt_template": prompt_template, "args": args})

    utils.log(f"Running {len(jobs)} RLM queries with {args.num_workers} workers (RLM max_timeout={args.per_instance_timeout}s, max_iterations={args.max_iterations}, model={args.model})")
    predictions = []
    with cf.ThreadPoolExecutor(max_workers=args.num_workers) as ex:
        futures = {ex.submit(run_one_instance, job): job for job in jobs}
        for fut in tqdm(cf.as_completed(futures), total=len(futures), desc="RLM prior-work"):
            rec = fut.result()
            predictions.append(rec)
            try:
                os.chdir(base_cwd)
            except Exception:
                pass
            with SAVE_LOCK:
                try:
                    utils.save_json(predictions, output_path, metadata=utils.update_metadata([], args), overwrite=True)
                except Exception as save_err:
                    utils.log(f"Save failed for record {rec['corpus_id']}: {type(save_err).__name__}: {save_err}")
            utils.log(f"Done {rec['corpus_id']}: pool={rec['candidate_pool_size']} model_preds={rec['num_model_predictions']} halluc={rec['num_hallucinated_ids']} t={rec['execution_time_seconds']}")

    total_hallucinated = sum(p["num_hallucinated_ids"] for p in predictions)
    utils.log(f"Saved {len(predictions)} predictions to {output_path}")
    utils.log(f"Total hallucinated corpus_ids dropped across all instances: {total_hallucinated}")


if __name__ == "__main__":
    main()

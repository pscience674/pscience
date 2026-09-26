"""RLM baseline for coauthor prediction. Loads the complete pre-cutoff corpus (H^{<t_p}) into a Python REPL and lets a Recursive Language Model rank likely future coauthors of the seed author. Exposes all pre-cutoff papers + a pubs_by_author index, so the LLM has the same information any other baseline does."""

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
from task_coauthor_prediction.dataset import create_evaluation_instances, get_preexisting_publications_for_author

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


def build_candidate_corpus(target_date, all_papers):
    """Return all pre-cutoff papers as compact records. This is the full corpus H^{<t_p} the LLM gets to reason over."""
    return [{
        "corpus_id": p["corpus_id"],
        "title": p.get("title") or "",
        "abstract": p.get("abstract") or "",
        "date": p["date"],
        "categories": p.get("categories") or [],
        "author_ids": [a.get("author_id") for a in (p.get("authors") or []) if isinstance(a, dict) and a.get("author_id")],
        "author_names": [a.get("name") or "" for a in (p.get("authors") or []) if isinstance(a, dict) and a.get("author_id")],
    } for p in all_papers if p["date"] < target_date]


def build_pubs_by_author(candidate_corpus):
    """Index author_id -> list of corpus_ids (each appearing in `candidate_corpus`)."""
    pubs = {}
    for paper in candidate_corpus:
        for aid in paper["author_ids"]:
            pubs.setdefault(aid, []).append(paper["corpus_id"])
    return pubs


def build_seed_history(seed_author_id, cutoff_date, sd2publications, all_papers_dict, num_recent_papers=30):
    """Recent publications of the seed with coauthor names/IDs visible — a focal subset of the corpus."""
    pubs = get_preexisting_publications_for_author(seed_author_id, cutoff_date, sd2publications, all_papers_dict)
    recent = pubs[-num_recent_papers:]
    out = []
    for cid in recent:
        p = all_papers_dict.get(cid)
        if p is None:
            continue
        coauthors = []
        for a in p.get("authors") or []:
            if isinstance(a, dict) and a.get("author_id") and a.get("author_id") != seed_author_id:
                coauthors.append({"author_id": a.get("author_id"), "name": a.get("name") or ""})
        out.append({"corpus_id": cid, "title": p.get("title") or "", "abstract": p.get("abstract") or "", "date": p.get("date") or "", "categories": p.get("categories") or [], "coauthors": coauthors})
    return out


def render_prompt(template, seed_author_id, target_date, top_n, n_corpus, n_authors):
    return template.format(top_n=top_n, target_date=target_date, seed_author_id=seed_author_id, n_corpus=n_corpus, n_authors=n_authors)


def _try_parse_list(text):
    text = text.strip()
    for parser in (ast.literal_eval, json.loads):
        try:
            obj = parser(text)
            if isinstance(obj, list):
                return obj
        except (ValueError, SyntaxError, json.JSONDecodeError):
            continue
    return None


def parse_final_answer(response_text, valid_author_ids):
    parsed = _try_parse_list(response_text)
    if parsed is None:
        match = FINAL_RE.search(response_text)
        if match is not None:
            parsed = _try_parse_list(match.group(1))
    if parsed is None:
        return [], 0
    valid_set = set(valid_author_ids)
    kept = [aid for aid in parsed if isinstance(aid, str) and aid in valid_set]
    hallucinated = len(parsed) - len(kept)
    return kept, hallucinated


def pad_predictions(predicted_ids, k, all_author_ids, exclude):
    if len(predicted_ids) >= k:
        return predicted_ids[:k]
    excluded = set(predicted_ids) | set(exclude)
    available = list(all_author_ids - excluded)
    needed = k - len(predicted_ids)
    extras = random.sample(available, min(needed, len(available)))
    return predicted_ids + extras


def write_pickle(payload):
    fd, path = tempfile.mkstemp(prefix="rlm_coauthor_", suffix=".pkl")
    os.close(fd)
    with open(path, "wb") as f:
        pickle.dump(payload, f)
    with PICKLE_PATHS_LOCK:
        PICKLE_PATHS.append(path)
    return path


def run_one_instance(job):
    instance, args = job["instance"], job["args"]
    cutoff_date = instance["date"]
    seed_author_id = instance["first_author_id"]
    candidate_corpus = job["candidate_corpus"]
    pubs_by_author = job["pubs_by_author"]
    valid_author_ids = set(pubs_by_author.keys()) - {seed_author_id}
    seed_history = build_seed_history(seed_author_id, cutoff_date, job["sd2publications"], job["all_papers_dict"])

    pickle_path = write_pickle({"corpus": candidate_corpus, "pubs_by_author": pubs_by_author, "seed_history": seed_history, "seed_author_id": seed_author_id, "target_date": cutoff_date})
    response_text = ""
    execution_time = None
    usage_summary = None

    try:
        setup_code = f"import pickle\n_payload = pickle.load(open({pickle_path!r}, 'rb'))\ncorpus = _payload['corpus']\npubs_by_author = _payload['pubs_by_author']\ncorpus_by_id = {{p['corpus_id']: p for p in corpus}}\nseed_history = _payload['seed_history']\nseed_author_id = _payload['seed_author_id']\ntarget_date = _payload['target_date']"
        env_kwargs = {"setup_code": setup_code}
        if args.environment == "ipython":
            env_kwargs["kernel_mode"] = args.ipython_kernel_mode
        rlm_kwargs = {"backend": args.backend, "backend_kwargs": {"model_name": args.model}, "environment": args.environment, "environment_kwargs": env_kwargs, "max_iterations": args.max_iterations, "max_depth": args.max_depth, "max_timeout": float(args.per_instance_timeout), "verbose": args.verbose}
        rlm = RLM(**rlm_kwargs)
        prompt = render_prompt(job["prompt_template"], seed_author_id, cutoff_date, args.top_n, len(candidate_corpus), len(pubs_by_author))
        result = rlm.completion(prompt)
        response_text = getattr(result, "response", "") or ""
        execution_time = getattr(result, "execution_time", None)
        usage_summary = getattr(result, "usage_summary", None)
        predicted_ids, hallucinated = parse_final_answer(response_text, valid_author_ids)
    except Exception as e:
        utils.log(f"Error on {instance['corpus_id']}: {type(e).__name__}: {e}")
        utils.log(traceback.format_exc())
        predicted_ids, hallucinated = [], 0

    num_model_predictions = len(predicted_ids)
    predicted_ids = pad_predictions(predicted_ids, args.k, valid_author_ids, [seed_author_id])
    predicted_scores = [1.0 / (i + 1) for i in range(len(predicted_ids))]

    return {"corpus_id": instance["corpus_id"], "first_author_id": seed_author_id, "gt_coauthor_ids": instance["gt_coauthor_ids"], "predicted_coauthor_ids": predicted_ids, "predicted_coauthor_scores": predicted_scores, "num_model_predictions": num_model_predictions, "num_hallucinated_ids": hallucinated, "corpus_size": len(candidate_corpus), "n_authors_in_corpus": len(pubs_by_author), "execution_time_seconds": execution_time, "usage_summary": str(usage_summary) if usage_summary is not None else None, "raw_response_truncated": response_text[-2000:]}


def main():
    parser = argparse.ArgumentParser(description="RLM baseline for coauthor prediction")
    parser.add_argument("--data_dir", type=str, default="data/prescience-ai", help="Dataset directory (with train/ and test/ subdirectories)")
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"], help="Dataset split to use")
    parser.add_argument("--output_dir", type=str, default="data/task_coauthor_prediction/test/predictions", help="Where to save predictions JSON")
    parser.add_argument("--model", type=str, default="gpt-5-mini", help="Model name passed to RLM backend")
    parser.add_argument("--backend", type=str, default="openai", help="RLM backend")
    parser.add_argument("--environment", type=str, default="ipython", choices=["local", "ipython"], help="RLM REPL environment")
    parser.add_argument("--ipython_kernel_mode", type=str, default="subprocess", choices=["in_process", "subprocess"], help="ipython kernel mode")
    parser.add_argument("--max_iterations", type=int, default=30, help="Max REPL turns")
    parser.add_argument("--max_depth", type=int, default=1, help="Max RLM recursion depth")
    parser.add_argument("--top_n", type=int, default=200, help="Max number of author_ids the LM should return")
    parser.add_argument("--k", type=int, default=1000, help="Pad/truncate predictions to this length")
    parser.add_argument("--num_workers", type=int, default=4, help="Concurrent instances")
    parser.add_argument("--per_instance_timeout", type=int, default=600, help="Seconds per RLM call")
    parser.add_argument("--seed_author_type", type=str, default="first", choices=["first", "last", "random", "highest_h_index"], help="Seed author selection strategy")
    parser.add_argument("--max_instances", type=int, default=None, help="Cap on instances")
    parser.add_argument("--verbose", action="store_true", help="Verbose RLM trajectories")
    parser.add_argument("--prompt_template", type=str, default="task_coauthor_prediction/templates/rlm_coauthor.prompt", help="Path to system prompt template")
    parser.add_argument("--output_suffix", type=str, default="", help="Optional suffix on output filename")
    args = parser.parse_args()

    abs_output_dir = os.path.abspath(args.output_dir)
    os.makedirs(abs_output_dir, exist_ok=True)
    output_path = os.path.join(abs_output_dir, f"predictions.rlm.{args.model}.{args.seed_author_type}{args.output_suffix}.json")
    utils.log(f"Output path resolved to absolute: {output_path}")

    utils.log(f"Loading corpus from {args.data_dir} (split={args.split})")
    all_papers, sd2publications, _ = utils.load_corpus(data_dir=args.data_dir, split=args.split, embedding_type=None, load_sd2publications=True)
    all_papers_dict = {p["corpus_id"]: p for p in all_papers}
    utils.log(f"Loaded {len(all_papers)} papers and {len(sd2publications)} author publication histories")

    base_cwd = os.getcwd()

    utils.log(f"Creating evaluation instances with seed_author_type={args.seed_author_type}")
    evaluation_instances = create_evaluation_instances(all_papers, sd2publications, all_papers_dict, args.seed_author_type)
    utils.log(f"Created {len(evaluation_instances)} evaluation instances")

    if args.max_instances is not None and args.max_instances < len(evaluation_instances):
        selected_indices = sorted(random.sample(range(len(evaluation_instances)), args.max_instances))
        utils.log(f"Randomly selected {len(selected_indices)} instances to evaluate")
    else:
        selected_indices = list(range(len(evaluation_instances)))

    with open(args.prompt_template, "r") as f:
        prompt_template = f.read()
    utils.log(f"Using prompt template: {args.prompt_template}")

    # Each instance has its own cutoff_date, so the pre-cutoff corpus differs per-instance.
    # We build candidate_corpus + pubs_by_author per instance below in run_one_instance via a
    # cached helper keyed on the cutoff date (so instances sharing a date reuse the same view).
    corpus_cache = {}
    corpus_cache_lock = threading.Lock()

    def get_corpus_for_cutoff(cutoff_date):
        with corpus_cache_lock:
            if cutoff_date not in corpus_cache:
                cc = build_candidate_corpus(cutoff_date, all_papers)
                pa = build_pubs_by_author(cc)
                corpus_cache[cutoff_date] = (cc, pa)
            return corpus_cache[cutoff_date]

    jobs = []
    for idx in selected_indices:
        date, instance = evaluation_instances[idx]
        instance = {**instance, "date": date}
        candidate_corpus, pubs_by_author = get_corpus_for_cutoff(date)
        jobs.append({"instance": instance, "candidate_corpus": candidate_corpus, "pubs_by_author": pubs_by_author, "all_papers_dict": all_papers_dict, "sd2publications": sd2publications, "prompt_template": prompt_template, "args": args})
    utils.log(f"Built candidate corpus for {len(corpus_cache)} unique cutoff dates")

    utils.log(f"Running {len(jobs)} RLM queries with {args.num_workers} workers (RLM max_timeout={args.per_instance_timeout}s, max_iterations={args.max_iterations}, model={args.model})")
    predictions = []
    with cf.ThreadPoolExecutor(max_workers=args.num_workers) as ex:
        futures = {ex.submit(run_one_instance, job): job for job in jobs}
        for fut in tqdm(cf.as_completed(futures), total=len(futures), desc="RLM coauthor"):
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
            utils.log(f"Done {rec['corpus_id']}: corpus={rec['corpus_size']} authors={rec['n_authors_in_corpus']} model_preds={rec['num_model_predictions']} halluc={rec['num_hallucinated_ids']} t={rec['execution_time_seconds']}")

    total_hallucinated = sum(p["num_hallucinated_ids"] for p in predictions)
    utils.log(f"Saved {len(predictions)} predictions to {output_path}")
    utils.log(f"Total hallucinated author_ids dropped across all instances: {total_hallucinated}")


if __name__ == "__main__":
    main()

"""RLM baseline for future-influential co-citation prediction. Loads the date-filtered candidate pool + the target paper's own key_references into a Python REPL and lets a Recursive Language Model rank papers most likely to be co-cited alongside the target in the future."""

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
from task_cocitation_prediction.dataset import create_evaluation_instances

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
    """Return the per-instance candidate pool: every paper strictly before target_date, excluding the target itself. Each candidate carries title/abstract/date/categories + the corpus_ids of its own key_references (so the LLM can reason over the citation graph)."""
    return [{
        "corpus_id": p["corpus_id"],
        "title": p.get("title") or "",
        "abstract": p.get("abstract") or "",
        "date": p["date"],
        "categories": p.get("categories") or [],
        "key_reference_corpus_ids": [r["corpus_id"] for r in (p.get("key_references") or []) if isinstance(r, dict) and "corpus_id" in r],
    } for p in all_papers if p["date"] < target_date and p["corpus_id"] != target_corpus_id]


def build_target_references(target_paper, all_papers_dict):
    """The target paper's own key references — visible to the RLM as the seed of the cocitation reasoning."""
    out = []
    for r in target_paper.get("key_references") or []:
        if not isinstance(r, dict):
            continue
        rid = r.get("corpus_id")
        ref_paper = all_papers_dict.get(rid) if rid is not None else None
        if ref_paper is None:
            continue
        out.append({
            "corpus_id": rid,
            "title": ref_paper.get("title") or "",
            "abstract": ref_paper.get("abstract") or "",
            "date": ref_paper.get("date") or "",
            "categories": ref_paper.get("categories") or [],
        })
    return out


def render_prompt(template, target_date, n_target_refs, top_n, candidate_pool_size, target_title):
    return template.format(top_n=top_n, target_date=target_date, n_target_refs=n_target_refs, candidate_pool_size=candidate_pool_size, target_title_preview=(target_title or "")[:120])


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


def parse_final_answer(response_text, valid_corpus_ids):
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
    if len(predicted_ids) >= k:
        return predicted_ids[:k]
    excluded = set(predicted_ids) | set(exclude)
    available = [cid for cid in all_papers_dict if cid not in excluded and all_papers_dict[cid]["date"] < cutoff_date]
    needed = k - len(predicted_ids)
    extras = random.sample(available, min(needed, len(available)))
    return predicted_ids + extras


def write_pickle(payload):
    fd, path = tempfile.mkstemp(prefix="rlm_cocitation_", suffix=".pkl")
    os.close(fd)
    with open(path, "wb") as f:
        pickle.dump(payload, f)
    with PICKLE_PATHS_LOCK:
        PICKLE_PATHS.append(path)
    return path


def run_one_instance(job):
    instance, args = job["instance"], job["args"]
    cutoff_date = instance["date"]
    target_corpus_id = instance["corpus_id"]
    target_paper = job["all_papers_dict"].get(target_corpus_id, {})
    candidate_pool = build_candidate_pool(target_corpus_id, cutoff_date, job["all_papers"])
    valid_ids = [c["corpus_id"] for c in candidate_pool]
    target_refs = build_target_references(target_paper, job["all_papers_dict"])
    target_title = target_paper.get("title") or ""
    target_abstract = target_paper.get("abstract") or ""
    target_topic_labels = target_paper.get("topic_labels") or []
    target_categories = target_paper.get("categories") or []

    pickle_path = write_pickle({"candidates": candidate_pool, "target_references": target_refs, "target_date": cutoff_date, "target_title": target_title, "target_abstract": target_abstract, "target_topic_labels": target_topic_labels, "target_categories": target_categories})
    response_text = ""
    execution_time = None
    usage_summary = None

    try:
        setup_code = f"import pickle\n_payload = pickle.load(open({pickle_path!r}, 'rb'))\ncandidates = _payload['candidates']\ncorpus_by_id = {{c['corpus_id']: c for c in candidates}}\ntarget_references = _payload['target_references']\ntarget_date = _payload['target_date']\ntarget_title = _payload['target_title']\ntarget_abstract = _payload['target_abstract']\ntarget_topic_labels = _payload['target_topic_labels']\ntarget_categories = _payload['target_categories']"
        env_kwargs = {"setup_code": setup_code}
        if args.environment == "ipython":
            env_kwargs["kernel_mode"] = args.ipython_kernel_mode
        rlm_kwargs = {"backend": args.backend, "backend_kwargs": {"model_name": args.model}, "environment": args.environment, "environment_kwargs": env_kwargs, "max_iterations": args.max_iterations, "max_depth": args.max_depth, "max_timeout": float(args.per_instance_timeout), "verbose": args.verbose}
        rlm = RLM(**rlm_kwargs)
        prompt = render_prompt(job["prompt_template"], cutoff_date, len(target_refs), args.top_n, len(candidate_pool), target_title)
        result = rlm.completion(prompt)
        response_text = getattr(result, "response", "") or ""
        execution_time = getattr(result, "execution_time", None)
        usage_summary = getattr(result, "usage_summary", None)
        predicted_ids, hallucinated = parse_final_answer(response_text, valid_ids)
    except Exception as e:
        utils.log(f"Error on {target_corpus_id}: {type(e).__name__}: {e}")
        utils.log(traceback.format_exc())
        predicted_ids, hallucinated = [], 0

    num_model_predictions = len(predicted_ids)
    predicted_ids = pad_predictions(predicted_ids, args.k, job["all_papers_dict"], cutoff_date, [target_corpus_id])
    predicted_scores = [1.0 / (i + 1) for i in range(len(predicted_ids))]

    return {"corpus_id": target_corpus_id, "gt_cocited_ids": instance["gt_cocited_ids"], "gt_cocited_counts": instance["gt_cocited_counts"], "predicted_cocited_ids": predicted_ids, "predicted_cocited_scores": predicted_scores, "num_model_predictions": num_model_predictions, "num_hallucinated_ids": hallucinated, "candidate_pool_size": len(candidate_pool), "execution_time_seconds": execution_time, "usage_summary": str(usage_summary) if usage_summary is not None else None, "raw_response_truncated": response_text[-2000:]}


def main():
    parser = argparse.ArgumentParser(description="RLM baseline for co-citation prediction")
    parser.add_argument("--data_dir", type=str, default="data/prescience-ai", help="Dataset directory (with train/ and test/ subdirectories)")
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"], help="Dataset split to use")
    parser.add_argument("--output_dir", type=str, default="data/task_cocitation_prediction/test/predictions", help="Where to save predictions JSON")
    parser.add_argument("--model", type=str, default="gpt-5-mini", help="Model name passed to RLM backend")
    parser.add_argument("--backend", type=str, default="openai", help="RLM backend")
    parser.add_argument("--environment", type=str, default="ipython", choices=["local", "ipython"], help="RLM REPL environment")
    parser.add_argument("--ipython_kernel_mode", type=str, default="subprocess", choices=["in_process", "subprocess"], help="ipython kernel mode")
    parser.add_argument("--max_iterations", type=int, default=30, help="Max REPL turns")
    parser.add_argument("--max_depth", type=int, default=1, help="Max RLM recursion depth")
    parser.add_argument("--top_n", type=int, default=200, help="Max number of corpus_ids the LM should return")
    parser.add_argument("--k", type=int, default=1000, help="Pad/truncate predictions to this length")
    parser.add_argument("--num_workers", type=int, default=4, help="Concurrent instances")
    parser.add_argument("--per_instance_timeout", type=int, default=600, help="Seconds per RLM call")
    parser.add_argument("--eval_months", type=int, default=4, help="Size of evaluation cohort window in months")
    parser.add_argument("--lookahead_months", type=int, default=8, help="Size of future-citation observation window in months")
    parser.add_argument("--eval_instances_cache", type=str, default=None, help="Path to cached evaluation instances JSON")
    parser.add_argument("--max_instances", type=int, default=None, help="Cap on instances")
    parser.add_argument("--verbose", action="store_true", help="Verbose RLM trajectories")
    parser.add_argument("--prompt_template", type=str, default="task_cocitation_prediction/templates/rlm_cocitation.prompt", help="Path to system prompt template")
    parser.add_argument("--output_suffix", type=str, default="", help="Optional suffix on output filename")
    args = parser.parse_args()

    abs_output_dir = os.path.abspath(args.output_dir)
    os.makedirs(abs_output_dir, exist_ok=True)
    output_path = os.path.join(abs_output_dir, f"predictions.rlm.{args.model}{args.output_suffix}.json")
    utils.log(f"Output path resolved to absolute: {output_path}")

    utils.log(f"Loading corpus from {args.data_dir} (split={args.split})")
    all_papers, _, _ = utils.load_corpus(data_dir=args.data_dir, split=args.split, embedding_type=None, load_sd2publications=False)
    all_papers_dict = {p["corpus_id"]: p for p in all_papers}
    utils.log(f"Loaded {len(all_papers)} papers")

    base_cwd = os.getcwd()

    utils.log("Creating evaluation instances")
    evaluation_instances = create_evaluation_instances(all_papers, all_papers_dict, eval_months=args.eval_months, lookahead_months=args.lookahead_months, cache_path=args.eval_instances_cache)
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
        jobs.append({"instance": instance, "all_papers": all_papers, "all_papers_dict": all_papers_dict, "prompt_template": prompt_template, "args": args})

    utils.log(f"Running {len(jobs)} RLM queries with {args.num_workers} workers (RLM max_timeout={args.per_instance_timeout}s, max_iterations={args.max_iterations}, model={args.model})")
    predictions = []
    with cf.ThreadPoolExecutor(max_workers=args.num_workers) as ex:
        futures = {ex.submit(run_one_instance, job): job for job in jobs}
        for fut in tqdm(cf.as_completed(futures), total=len(futures), desc="RLM cocitation"):
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

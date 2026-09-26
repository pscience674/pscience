"""Claude baseline for followup work prediction. Generates title/abstract predictions using Claude API."""

import os
import argparse
import concurrent.futures as cf

import anthropic
from tqdm import tqdm

import utils
from task_followup_prediction.dataset import get_query_papers, create_messages


def query_claude(record):
    """Query Claude API and parse the response into title, abstract, and reasoning."""
    client, model, messages = record["client"], record["model"], record["messages"]
    del record["client"], record["model"], record["messages"]
    system_prompt = messages[0]["content"]
    messages = messages[1:]

    max_tokens = record["max_tokens"]
    del record["max_tokens"]

    try:
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            # Cache the shared system prompt; the per-paper user message varies. Prefixes below a model's
            # minimum cacheable length (e.g. 4096 tokens on Opus 4.6) silently aren't cached.
            system=[{"type": "text", "text": system_prompt, "cache_control": {"type": "ephemeral"}}],
            messages=messages
        )
        u = response.usage
        utils.log(f"usage model={model} in={u.input_tokens} cache_read={u.cache_read_input_tokens} cache_write={u.cache_creation_input_tokens} out={u.output_tokens}")
        if response.stop_reason == "refusal":
            utils.log(f"Refusal on corpus_id {record['corpus_id']}")
            record.update({"title": "", "abstract": "", "reasoning": "", "refusal": True})
            return record
        answer = next(b.text for b in response.content if b.type == "text").strip()

        reasoning = answer.split("Reasoning: ")[1].split("Title: ")[0].strip()
        title = answer.split("Title: ")[1].split("Abstract: ")[0].strip()
        abstract = answer.split("Abstract: ")[1].strip()
        record.update({
            "title": title,
            "abstract": abstract,
            "reasoning": reasoning
        })
        return record

    except Exception as e:
        utils.log(f"Error: {e}")
        record.update({"title": "", "abstract": "", "reasoning": ""})
        return record


def main():
    """Generate followup work predictions using Claude API."""
    parser = argparse.ArgumentParser(description="Claude baseline for followup work prediction")
    parser.add_argument("--data_dir", type=str, default="data/prescience-ai", help="Dataset directory (with train/ and test/ subdirectories)")
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"], help="Dataset split")
    parser.add_argument("--output_dir", type=str, default="data/task_followup_prediction/test/generations")
    parser.add_argument("--model", type=str, default="claude-3-5-sonnet-20241022")
    parser.add_argument("--max_query_papers", type=int, default=5000)
    parser.add_argument("--num_workers", type=int, default=16, help="Number of parallel requests")
    parser.add_argument("--max_tokens", type=int, default=4096, help="Max output tokens per generation (raise for models with always-on thinking)")
    parser.add_argument("--system_prompt_path", type=str, default="task_followup_prediction/templates/prediction_system.prompt", help="Path to the system prompt (use a domain-specific prompt for non-AI corpora)")
    args = parser.parse_args()

    all_papers, _, _ = utils.load_corpus(data_dir=args.data_dir, split=args.split, embedding_type=None, load_sd2publications=False)
    all_papers_dict = {p["corpus_id"]: p for p in all_papers}
    query_papers = get_query_papers(all_papers, max_papers=args.max_query_papers)

    with open(args.system_prompt_path, "r") as f:
        system_prompt = f.read()

    anthropic_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    utils.log(f"Loaded {len(all_papers)} papers, {len(query_papers)} query papers after filtering")

    utils.log("Preparing jobs")
    jobs = []
    for rec in query_papers:
        rec["client"] = anthropic_client
        rec["model"] = args.model
        rec["max_tokens"] = args.max_tokens
        rec["messages"] = create_messages(rec, system_prompt, all_papers_dict)
        jobs.append(rec)

    utils.log(f"Running {len(jobs)} queries with {args.num_workers} workers")
    with cf.ThreadPoolExecutor(max_workers=args.num_workers) as ex:
        results_iter = ex.map(query_claude, jobs)
        for rec in tqdm(results_iter, total=len(jobs), desc="Generating samples"):
            pass

    output_path = os.path.join(args.output_dir, f"generations.{args.model}.json")
    utils.save_json(query_papers, output_path, overwrite=True, metadata=utils.update_metadata([], args))
    utils.log(f"Saved {len(query_papers)} generations to {output_path}")


if __name__ == "__main__":
    main()

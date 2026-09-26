# PreScience

**A Benchmark for Scientific Forecasting** (anonymized code release for double-blind review)

> **Note:** We lightly refactored this code for the anonymized submission (e.g., loading the dataset from a local directory and removing internal infrastructure). We have not yet thoroughly tested the anonymized version, but we believe it works end to end.

## Overview

Can AI systems trained on the existing scientific record forecast the advances that will follow? PreScience is a dataset and benchmark for scientific forecasting built around 147K recent arXiv papers from two domains, AI (PreScience-AI) and quantum physics (PreScience-Quantum), together with companion papers covering author publication histories and citation links (1.06M papers in total). Paper records include titles, abstracts, disambiguated author identities, influential references, topic labels (PreScience-AI), citation trajectories, and metadata snapshotted to respect temporal cutoffs. We instantiate seven tasks: five paper-anchored tasks (contribution generation, collaborator prediction, prior work selection, citation count prediction, and future combination prediction) and two aggregate topic trend forecasting variants.

| Task | Description | Metrics |
|------|-------------|---------|
| Collaborator Prediction | Predict the remaining authors on a future paper given a seed author | nDCG, R-Precision |
| Prior Work Selection | Predict the influential references of a future paper given its authors | nDCG, R-Precision |
| Contribution Generation | Generate a paper's title and abstract given its influential references | LACER, ROUGE-L, BERTScore |
| Citation Count Prediction | Predict a paper's 12-month cumulative citation count | MAE, Pearson, Spearman |
| Future Combination Prediction | Rank prior papers by how often they will be jointly cited as influential references with a target paper in future work | nDCG, R-Precision |
| Topic Trend Forecasting (Unary) | Predict the change in each topic's share of papers between train and test periods | R², Pearson |
| Topic Trend Forecasting (Pair) | Predict the change in each topic-pair's share between train and test periods | R², Pearson |

## Dataset

Download the dataset archive (link in the paper) and unpack it so that each domain has `train/` and `test/` subdirectories:

```
data/
├── prescience-ai/        # PreScience-AI: cs.CL, cs.LG, cs.AI, cs.CV, cs.IR, cs.NE
│   ├── train/            # target papers Aug 2024 – Jul 2025
│   │   ├── all_papers.json
│   │   └── sd2publications.json
│   └── test/             # target papers Aug 2025 – Jul 2026
└── prescience-quantum/   # PreScience-Quantum: quant-ph, cond-mat, physics.optics, physics.atom-ph
    ├── train/
    └── test/
```

The files are distributed gzipped; run `gunzip -r data/` after unpacking. All scripts take `--data_dir <domain dir>` (default `data/prescience-ai`); pass `--data_dir data/prescience-quantum` for PreScience-Quantum. Each `all_papers.json` holds every paper in the split (target and companion papers, distinguished by the `roles` field); `sd2publications.json` maps disambiguated author IDs to their publication corpus IDs.

Embeddings are not included because of their size (~30 GB); compute them with `dataset/embeddings/compute_paper_embeddings.py` (see [Computing Embeddings](#computing-embeddings)).

## Setup

```bash
cd prescience
pip install -r requirements.txt
```

**Prerequisites:**
- Python 3.10+
- GPU with 24 GB+ VRAM for GritLM-7B embeddings and 8B model fine-tuning (BF16, no quantization); CUDA 12 required (`faiss-gpu-cu12`)
- Embedding-based task baselines and `multiturn/simulate.py` require precomputed embeddings
- `export OPENAI_API_KEY=<your_key>` — required for GPT baselines, LACER evaluation, and gold paraphrase generation
- `export ANTHROPIC_API_KEY=<your_key>` — required for Claude baselines
- SPECTER2 has dependency conflicts with the main environment; use `requirements_specter2.txt` in a separate environment
- OLMo 3 fine-tuning requires `transformers>=4.57.0`, which conflicts with GritLM; use `requirements_olmo3.txt` in a separate environment


## Codebase Structure

```
prescience/
├── task_coauthor_prediction/     # Collaborator prediction baselines, evaluation and analyses
├── task_priorwork_prediction/    # Prior work selection baselines, evaluation and analyses
├── task_followup_prediction/
│   ├── generate/                 # Contribution generation baselines (GPT, Claude, OLMo, LLaMA)
│   └── evaluate/                 # LACERScore, BERTScore, ROUGE-L evaluation
│   └── analysis/
├── task_impact_prediction/       # Impact prediction baselines, evaluation and analyses
├── multiturn/
│   ├── simulate.py               # End-to-end multi-turn simulation
│   └── analysis/                 # Simulation analysis scripts
├── dataset/
│   ├── corpus/                   # 7-stage corpus creation pipeline
│   ├── embeddings/               # Embedding computation
│   ├── dataset_stats.py          # Dataset statistics table
│   └── s2and_prep/               # S2AND author disambiguation features
├── utils.py                      # Shared utilities (I/O, embeddings, FAISS, S2 API)
└── requirements.txt
```


## Running Baselines

All baseline scripts save predictions to `data/<task_dir>/<split>/predictions/` and generation scripts save to `data/task_followup_prediction/<split>/generations/`, which are the inputs to the evaluation scripts.

### Collaborator Prediction

```bash
# Frequency
python3 -m task_coauthor_prediction.baseline_frequency --split test

# Rank Fusion (requires precomputed embeddings)
python3 -m task_coauthor_prediction.baseline_rank_fusion --split test --embeddings_dir data/corpus/test --embedding_type grit

# Embedding Fusion
python3 -m task_coauthor_prediction.baseline_embedding_fusion --split test --embeddings_dir data/corpus/test --embedding_type grit

# Hierarchical Clustering
python3 -m task_coauthor_prediction.baseline_hierarchical --split test --embeddings_dir data/corpus/test --embedding_type grit

# Embedding Fusion Projected (train projection, then run inference)
python3 -m task_coauthor_prediction.train_projection --split train --embeddings_dir data/corpus/train --embedding_type grit --output_dir data/task_coauthor_prediction/train/projection_models
python3 -m task_coauthor_prediction.baseline_mean_pooling_projected --split test --embeddings_dir data/corpus/test --embedding_type grit --checkpoint data/task_coauthor_prediction/train/projection_models/projection.grit.pt

# Evaluate
python3 -m task_coauthor_prediction.evaluate --predictions_path data/task_coauthor_prediction/test/predictions/predictions.frequency.one_shot.first.json
```

Table 3 results are reported across `--embedding_type` values: `gtr`, `specter2`, `grit`. The `--seed_author_type` flag defaults to `first` (matching Table 3); other options are `last`, `random`, `highest_h_index`.

The `task_coauthor_prediction/analysis/` directory contains the scripts used to generate the paper's analysis plots (e.g., nDCG vs. experience, R-Precision by familiarity). These expect the full set of baseline predictions with `grit` embeddings and may not run with partial inputs.

### Prior Work Selection

```bash
# Frequency
python3 -m task_priorwork_prediction.baseline_frequency --split test

# Embedding Fusion (mean pool of authored papers; "Emb. Fusion" in Table 3)
python3 -m task_priorwork_prediction.baseline_mean_pooling --split test --embeddings_dir data/corpus/test --embedding_type grit

# Embedding Fusion Refs (mean pool of cited references; "Emb. Fusion Refs" in Table 3)
python3 -m task_priorwork_prediction.baseline_embedding_fusion --split test --embeddings_dir data/corpus/test --embedding_type grit

# Rank Fusion
python3 -m task_priorwork_prediction.baseline_rank_fusion --split test --embeddings_dir data/corpus/test --embedding_type grit

# Hierarchical Clustering
python3 -m task_priorwork_prediction.baseline_hierarchical --split test --embeddings_dir data/corpus/test --embedding_type grit

# Embedding Fusion Projected (train projection, then run inference)
python3 -m task_priorwork_prediction.train_projection --split train --embeddings_dir data/corpus/train --embedding_type grit --output_dir data/task_priorwork_prediction/train/projection_models
python3 -m task_priorwork_prediction.baseline_mean_pooling_projected --split test --embeddings_dir data/corpus/test --embedding_type grit --checkpoint data/task_priorwork_prediction/train/projection_models/projection.grit.pt

# Evaluate
python3 -m task_priorwork_prediction.evaluate --predictions_path data/task_priorwork_prediction/test/predictions/predictions.frequency.json
```

Table 3 results are reported across `--embedding_type` values: `gtr`, `specter2`, `grit`.

The `task_priorwork_prediction/analysis/` directory contains the scripts used to generate the paper's analysis plots (e.g., nDCG vs. experience, R-Precision by familiarity). These expect the full set of baseline predictions with `grit` embeddings and may not run with partial inputs.

### Contribution Generation

```bash
# GPT models (requires OPENAI_API_KEY)
python3 -m task_followup_prediction.generate.baseline_gpt_parallel --split test --model gpt-4o-2024-11-20

# Claude models (requires ANTHROPIC_API_KEY)
python3 -m task_followup_prediction.generate.baseline_claude_parallel --split test --model claude-sonnet-4-5-20250929

# Local models — vanilla
python3 -m task_followup_prediction.generate.baseline_local --split test --model llama3.1-8b

# Local models — fine-tuned (LoRA)
python3 -m task_followup_prediction.generate.baseline_local --split test --model llama3.1-8b --adapter_path data/task_followup_prediction/train/lora_models/llama3.1-8b/final/adapter

# Same-topic baseline
python3 -m task_followup_prediction.generate.baseline_same_topic --split test

# Key reference baseline
python3 -m task_followup_prediction.generate.baseline_key_reference --split test

# Gold paraphrase (requires OPENAI_API_KEY)
python3 -m task_followup_prediction.generate.baseline_paraphrased_gold --split test --model gpt-5-2025-08-07
```

**Model name mapping** (paper name → `--model` value):

| Paper Name | `--model` | Script |
|---|---|---|
| GPT 4o | `gpt-4o-2024-11-20` | `baseline_gpt_parallel` |
| GPT 4.1 | `gpt-4.1-2025-04-14` | `baseline_gpt_parallel` |
| GPT o3 | `o3-2025-04-16` | `baseline_gpt_parallel` |
| GPT 5 | `gpt-5-2025-08-07` | `baseline_gpt_parallel` |
| GPT 5.1 | `gpt-5.1-chat-latest` | `baseline_gpt_parallel` |
| GPT 5.2 | `gpt-5.2-2025-12-11` | `baseline_gpt_parallel` |
| Claude Sonnet 4.5 | `claude-sonnet-4-5-20250929` | `baseline_claude_parallel` |
| Claude Opus 4.5 | `claude-opus-4-5-20251101` | `baseline_claude_parallel` |
| LLaMA 3.1 8B | `llama3.1-8b` | `baseline_local` |
| OLMo 3 7B | `olmo3-7b` | `baseline_local` |

Table 2 reports only fine-tuned (FT) results for LLaMA and OLMo.

**LoRA fine-tuning:**

OLMo 3 fine-tuning requires a separate environment due to transformers version conflicts with GritLM; install with `pip install -r requirements_olmo3.txt`.

```bash
# Hyperparameter search + final training
python3 -m task_followup_prediction.generate.finetune_lora --input_dir data/corpus/train --output_dir data/task_followup_prediction/train/lora_models --model llama3.1-8b --n_trials 20 --val_ratio 0.15 --num_train_epochs 3

# Skip HP search with known-good params
python3 -m task_followup_prediction.generate.finetune_lora --input_dir data/corpus/train --output_dir data/task_followup_prediction/train/lora_models --model llama3.1-8b --skip_hp_search --learning_rate 2e-4 --lora_r 16
```

**Evaluate:**

```bash
# LACERScore (default judge: gpt-5-2025-08-07; also supports Anthropic via --judge anthropic)
python3 -m task_followup_prediction.evaluate.lacer_score --input_path <generations_file>

# BERTScore
python3 -m task_followup_prediction.evaluate.bert_score --input_path <generations_file>

# ROUGE-L
python3 -m task_followup_prediction.evaluate.rouge_score --input_path <generations_file>
```

The `task_followup_prediction/analysis/` directory contains the scripts used to generate the paper's analysis plots (e.g., LACER by category, LACER vs. citations). These require LACER-scored generation files as input.

### Impact Prediction

**Feature-to-flag mapping** (for understanding Table 4 feature set combinations):

| Paper Feature Set | CLI Flags |
|---|---|
| Target Text | `--use_followup_work_paper` |
| Context Text | `--use_author_papers --use_prior_work_papers` |
| Bibliometrics | `--use_author_numbers --use_prior_work_numbers` |

```bash
# Ridge — Target Text (GRIT)
python3 -m task_impact_prediction.baseline_embedding_ridge --split test --train_embeddings_dir data/corpus/train --test_embeddings_dir data/corpus/test --embedding_type grit --use_followup_work_paper

# XGBoost — Target Text only (GRIT)
python3 -m task_impact_prediction.baseline_xgboost_regressor --split test --train_embeddings_dir data/corpus/train --test_embeddings_dir data/corpus/test --embedding_type grit --use_followup_work_paper

# XGBoost — Target + Context Text
python3 -m task_impact_prediction.baseline_xgboost_regressor --split test --train_embeddings_dir data/corpus/train --test_embeddings_dir data/corpus/test --embedding_type grit --use_followup_work_paper --use_author_papers --use_prior_work_papers

# XGBoost — Target + Context + Bibliometrics (full model)
python3 -m task_impact_prediction.baseline_xgboost_regressor --split test --train_embeddings_dir data/corpus/train --test_embeddings_dir data/corpus/test --embedding_type grit --use_followup_work_paper --use_author_papers --use_prior_work_papers --use_author_numbers --use_prior_work_numbers

# XGBoost — Bibliometrics only
python3 -m task_impact_prediction.baseline_xgboost_regressor --split test --train_embeddings_dir data/corpus/train --test_embeddings_dir data/corpus/test --embedding_type grit --use_author_numbers --use_prior_work_numbers

# Evaluate
python3 -m task_impact_prediction.evaluate --predictions_path data/task_impact_prediction/test/predictions/predictions.ridge_grit.json
```

Replace `grit` with `gtr` or `specter2` to reproduce other embedding rows in Table 4.

The `task_impact_prediction/analysis/` directory contains the scripts used to generate the paper's analysis plots (e.g., prediction scatter, SHAP analysis). These require evaluation output files as input.


## Computing Embeddings

Embedding-based baselines require precomputed embeddings. Compute them for each split you need:

```bash
python3 -m dataset.embeddings.compute_paper_embeddings --split test --embedding_type grit --output_dir data/corpus/test
python3 -m dataset.embeddings.compute_paper_embeddings --split train --embedding_type grit --output_dir data/corpus/train
```

| Type | Model | Dimension |
|------|-------|-----------|
| `gtr` | sentence-transformers/gtr-t5-large | 768 |
| `specter2` | allenai/specter2_base | 768 |
| `grit` | GritLM/GritLM-7B | 4096 |

SPECTER2 embeddings require a separate environment; install with `pip install -r requirements_specter2.txt`.


## Multi-turn Simulation

The simulation composes collaborator prediction, prior work selection, and contribution generation into a pipeline that generates a synthetic corpus day-by-day over a specified time horizon.

```bash
python3 -m multiturn.simulate --calibration_split train --embeddings_dir data/corpus/train \
  --author_embedding_type grit --paper_embedding_type grit \
  --coauthor_baseline embedding_fusion --priorwork_baseline embedding_fusion \
  --generation_backend openai --generation_model gpt-5-2025-08-07 \
  --output_dir data/multiturn/simulated --depth 365 --seed 42
```

Key arguments:
- `--coauthor_baseline`: `frequency`, `embedding_fusion`, `rank_fusion`
- `--priorwork_baseline`: `frequency`, `embedding_fusion`, `rank_fusion`
- `--generation_backend`: `openai`, `anthropic`
- `--generation_model`: any supported model name (e.g., `gpt-5-2025-08-07`)
- `--seed`: random seed for reproducibility

The paper uses `embedding_fusion` for both collaborator and prior work prediction with GPT-5 for generation, runs the simulation 6 times with different `--seed` values, and reports 95% confidence intervals.

**Analysis.** The analysis pipeline has two stages: compute metrics, then plot. For example, to analyze author diversity:

```bash
# Compute diversity for natural and synthetic corpora
python3 -m multiturn.analysis.compute_author_diversity --compute_on natural --role target --output_path data/multiturn/analysis/author_diversity_natural.json
python3 -m multiturn.analysis.compute_author_diversity --compute_on synthetic --synthetic_dir data/multiturn/simulated --role synthetic --output_path data/multiturn/analysis/author_diversity_synthetic.json

# Plot
python3 -m multiturn.analysis.plot_author_diversity --natural_path data/multiturn/analysis/author_diversity_natural.json --synthetic_paths data/multiturn/analysis/author_diversity_synthetic.json
```

Other available analysis scripts follow the same compute→plot pattern:

| Analysis | Compute Script | Plot Script |
|----------|---------------|-------------|
| Author diversity | `compute_author_diversity` | `plot_author_diversity` |
| Key reference diversity | `compute_key_references_diversity` | `plot_key_references_diversity` |
| LACER diversity | `compute_diversity_neighbors` → `compute_lacer_diversity` | `plot_lacer_diversity` |
| LACER novelty | `compute_novelty_neighbors` → `compute_lacer_novelty` | `plot_lacer_novelty` |
| Topic distribution | `classify_synthetic_primary_categories` | `plot_topic_distribution` |


## Reproducing Paper Results

| Paper Reference | Tasks / Section | How to Reproduce |
|---|---|---|
| Table 3 | Collaborator Prediction + Prior Work Selection | Run all embedding baselines (Frequency, Rank Fusion, Emb. Fusion, Hier. Clustering, Projection) × 3 embedding types (`gtr`, `specter2`, `grit`) |
| Table 2 | Contribution Generation | Run all LLM baselines (see model mapping table) and evaluate with LACERScore, BERTScore, and ROUGE-L |
| Table 4 | Impact Prediction | Run XGBoost with feature combinations (Target Text, Context Text, Bibliometrics) × 3 embedding types |
| Figure 2 | Corpus Generation (Section 4.5) | Run `multiturn.simulate` with `--coauthor_baseline embedding_fusion --priorwork_baseline embedding_fusion --generation_model gpt-5-mini-2025-08-07`, 3 seeds |


## Building the Dataset from Scratch

The scripts below rebuild the corpus with the Semantic Scholar and arXiv APIs.

To rebuild from scratch, you need:
- **S2 API key**: Get one at https://www.semanticscholar.org/product/api#api-key and set `export S2_API_KEY=<your_key>`
- **arXiv snapshot**: Download from [Kaggle](https://www.kaggle.com/datasets/Cornell-University/arxiv) to `data/arxiv_snapshot/arxiv-metadata-oai-snapshot.json`
- **S2AND**: For author disambiguation (Stage 4), follow [S2AND setup](https://github.com/allenai/S2AND)

<details>
<summary><b>Full pipeline commands</b></summary>

Run each stage for both train and test splits. Commands below show the train split; for test, replace `2024-08-01`/`2025-08-01` with `2025-08-01`/`2026-08-01` and `train` with `test`.

```bash
# Stage 1: Download target papers from arXiv snapshot + S2 API
python3 -m dataset.corpus.download_target_papers --start_date 2024-08-01 --end_date 2025-08-01 --output_dir data/corpus/train

# Stage 2: Add key (highly influential) references
python3 -m dataset.corpus.add_key_references --input_dir data/corpus/train --output_dir data/corpus/train

# Stage 3: Add author rosters and publication histories
python3 -m dataset.corpus.add_authors --input_dir data/corpus/train --output_dir data/corpus/train

# Stage 4a: Prepare S2AND input features
python3 -m dataset.s2and_prep.download_s2and_features --input_dirs data/corpus/train data/corpus/test --output_dir data/corpus/s2and_prescience

# Stage 4b: Run S2AND clustering (see https://github.com/allenai/S2AND)

# Stage 4c: Merge disambiguated author identities into corpus
python3 -m dataset.corpus.merge_authors_in_corpus --input_dir data/corpus/train --output_dir data/corpus/train --s2and_data_dir data/corpus/s2and_prescience

# Stage 5: Add citation metadata (citation counts, h-index, trajectories)
python3 -m dataset.corpus.add_citation_metadata --input_dir data/corpus/train --output_dir data/corpus/train

# Stage 6: Replace titles/abstracts with official arXiv versions
python3 -m dataset.corpus.replace_title_abstracts_using_snapshot --input_dir data/corpus/train --output_dir data/corpus/train

# Stage 6b: Clean author names, drop temporally invalid key references and orphaned papers, then validate
python3 -m dataset.corpus.clean_corpus --input_path data/corpus/train/all_papers.stage06.json --output_path data/corpus/train/all_papers.json --sd2publications_path data/corpus/train/sd2publications.json
python3 -m dataset.corpus.validate_invariants --input_path data/corpus/train/all_papers.json

# Stage 7: Assign topic labels to target papers and attach them to the corpus
python3 -m dataset.corpus.assign_topics --splits train --topics_path dataset/corpus/topics_list_v5.txt --model gpt-5-2025-08-07 --output_path data/task_topic_growth_prediction/topic_labels.v5.gpt-5.4.full.jsonl
python3 -m dataset.corpus.add_topics --input_dir data/corpus/train --output_dir data/corpus/train --topics_path data/task_topic_growth_prediction/topic_labels.v5.gpt-5.4.full.jsonl --overwrite

# Stage 8: Compute embeddings
python3 -m dataset.embeddings.compute_paper_embeddings --split train --embedding_type gtr --output_dir data/corpus/train
python3 -m dataset.embeddings.compute_paper_embeddings --split train --embedding_type specter2 --output_dir data/corpus/train
python3 -m dataset.embeddings.compute_paper_embeddings --split train --embedding_type grit --output_dir data/corpus/train
```

</details>


## License

Code is licensed under Apache 2.0 — see [LICENSE](LICENSE). The dataset is released under [ODC-BY 1.0](https://opendatacommons.org/licenses/by/1.0/).

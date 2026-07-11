# Retrieval annotation pipeline

This pipeline keeps candidate discovery, human judgement, golden-set creation,
and evaluation as separate stages. Candidate generation never treats answer
keywords or retrieval scores as relevance labels.

## 1. Validate provenance

```powershell
python scripts/validate_retrieval_manifest.py
```

The validator checks every source file SHA-256 plus the configured corpus,
chunk, embedding, and retrieval versions. If an independent index/corpus hash
is available, compare it as well:

```powershell
python scripts/validate_retrieval_manifest.py --corpus-hash <sha256>
```

## 2. Generate candidates

```powershell
python scripts/generate_retrieval_golden.py --top-k 20 --timeout 60
```

The default output is
`evals/annotations/retrieval_candidates_v1.jsonl`. Each query contains the
stable union of Dense, BM25, and Hybrid candidates. Every candidate records
the full chunk, source metadata, per-route ranks/scores, and data/model
versions. `review_status` remains `pending`; this file is not a golden set.

A failed or timed-out query stops the run after flushing all completed rows.
Retry without duplicating those rows:

```powershell
python scripts/generate_retrieval_golden.py --resume --timeout 60
```

Use `--max-queries` for a bounded smoke run and `--output` for an alternate
annotation batch. Logs contain only the case ID and exception type, never raw
provider exception text, credentials, or signed request URLs.

## 3. Review labels

Write reviewed judgements to
`evals/annotations/retrieval_labels_v1.jsonl`. Grades are:

- `0`: irrelevant
- `1`: related but does not answer
- `2`: partially relevant
- `3`: directly relevant

Each case must have `review_status: "reviewed"`, a non-empty `reviewed_by`, and
at least one positive label. Keep a rationale for every grade, including zero.

## 4. Build the golden set

```powershell
python scripts/split_retrieval_labels.py
```

The script rejects pending, empty, duplicate, out-of-range, or structurally
invalid labels. It reads the fixed `split_seed` from the manifest and writes
`evals/retrieval_golden.jsonl` atomically. Each row has a deterministic `dev`
or `test` split, positive `relevant_doc_ids`, and the complete four-grade
`relevance` mapping used by nDCG.

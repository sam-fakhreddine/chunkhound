# Indexing Performance Results

Benchmarks, profiles, and correctness evidence for the batch-indexing
performance work (issue chunkhound/chunkhound#262).

## Headline

| Scenario (cold index, 4-core container) | Baseline (f322bfa) | After (this branch) |
|---|---|---|
| 3,994 files, parse+store only | 119.9 s | **86.0 s (−28%)** |
| 3,994 files **with embeddings** (dims=256, offline provider) | **DNF — DB grew past 7 GB mid-embed, run aborted by disk guard** | **completes: 666.9 s, peak DB 554 MB** |
| same at dims=1536 | **DNF — DB hit 27.4 GB (~39× data), disk exhausted, DuckDB fatal `InvalidatedDatabase`** | linear growth (run progressed healthily; see notes) |
| Tier-2 subset (640 files, 18,267 embeddings) | 120.0 s, peak DB 751 MB | **101.5 s, peak DB ~150 MB** |
| post-storage embedding sweep (Tier 2) | 97.2 s | **0.016 s** (work overlapped into storage) |

Search correctness: fixed-query top-10 snapshots (files, symbols, lines,
scores) are **byte-identical** between baseline and the final build; the
guardrail suite (recall anchors, chunks==embeddings, incremental semantics,
exclude filtering, close/reopen) passes at every increment.

## Setup

- Container: Linux, 4 CPU cores, 15 GB RAM, uv-managed env, Python 3.11
- DuckDB (python) 1.4.4, VSS extension
- Corpora (pinned):
  - **Tier 1** — Django 5.0 sdist (`pip download Django==5.0 --no-binary :all:`),
    3,994 indexable files → ~115k chunks. Run with `--no-embeddings`
    (isolates discovery/parse/store).
  - **Tier 2** — the sdist's `django/` package without `django/contrib`
    (849 files, 640 indexed → 18,267 chunks), with embeddings, dims=256.
  - **Full-pipeline** — the whole sdist with embeddings (the configuration
    the baseline cannot finish).
- Embeddings: deterministic offline provider (character n-gram hash vectors,
  computed off the event loop, no network) — the numbers measure the
  pipeline, not a model. Same text → same vector before/after, so search
  results are directly comparable.
- Harness: `scripts/bench_indexing.py` — drives the production path
  (`configure_registry` → `IndexingCoordinator` →
  `DirectoryIndexingService.process_directory`), fresh subprocess + fresh DB
  dir per run; reports wall, per-phase wall, peak RSS, final and **peak** DB
  size; dumps fixed-query top-K search snapshots for cross-build diffing;
  aborts on a runaway DB (disk guard).
- Multi-run rows are the **median of 3**; the full-pipeline completion runs
  are single runs (≈11–12 min each).
- Noise: files ≥128 KB flirt with the default 3 s per-file parse timeout, so
  totals vary ~±1.5% run-to-run (3,994–3,996 files).

## Baseline anatomy (verified against issue #262's hypotheses)

1. **Single-threaded DB writer** (`serial_executor.py`,
   `ThreadPoolExecutor(max_workers=1)`) — confirmed, with nuance: parsing
   already streamed, but storing one file cost 5–7 serialized executor
   dispatches (~0.3–0.7 ms each) in its own transaction: begin / SELECT file
   / INSERT-or-UPDATE / SELECT existing chunks / [delete] / insert / commit.
2. **Redundant SELECTs per write** — confirmed exactly: the coordinator's
   existence SELECT, then a second one inside `_executor_insert_file` for
   new files — all triply redundant on cold indexes because change detection
   had already batch-fetched every file's metadata.
3. **Embedding gated on 100% storage** — confirmed
   (`DirectoryIndexingService`: store → compact → embed → compact → HNSW).
4. **Per-write junk accumulation** — confirmed and dominant. Every ~300-row
   embedding batch committed with `force_checkpoint=True`; the embedding
   table's HNSW index was live through the whole cold-index embed phase
   (created at table creation — the existing drop-before-bulk ran before the
   table existed, and the final ensure no-op'd in 12 ms); DuckDB VSS
   re-serializes the whole index on every checkpoint ⇒ **O(N²) bytes
   written**. Isolation experiment (20×300 vectors, 6.1 MB raw): live index
   + per-batch checkpoint → 82.8 MB file; no index → 6.6 MB; index with one
   final checkpoint → 12.9 MB.
5. py-spy (60 s, Tier-2 embed): `serial-db` thread active in 67% of wall
   samples (71% of that in embedding upserts, ~10% in forced checkpoints);
   event loop 60% in `fnmatch` inside an embed pre-scan that loads every
   chunk (with code) twice.

## Increments

Each increment: separate commit, guardrails + smoke green, benchmarks re-run.

### A — skip pre-answered existence SELECTs (`ce81a24`)

Files proven absent by the change-detection pre-scan store with a single
INSERT (no existence SELECTs); a duplicate-key fallback keeps a stale
assumption exactly equivalent to the legacy update+diff path.
**Delta: within noise** (~1 ms/file); groundwork for B.

### B — one transaction + batched statements per parsed batch (`075a056`)

`prepare_files_batch_async` upserts every file row and returns existing
chunks in ONE dispatch; deletes collapse to one call per batch; chunk ids
are preallocated from the sequence so id↔chunk alignment no longer depends
on `RETURNING` order (which `preserve_insertion_order=false` — set by
bulk-optimized mutations on the same connection — would silently break).
Any batch failure falls back to the per-file path.
Also fixes a latent realtime bug: embedding ids were zipped against a
longer list (all parsed chunks), so partial-diff edits raised a length
mismatch and got no embeddings until a later sweep.

| Tier | Wall | parse_store phase |
|---|---|---|
| 1 | 120.0 → **88.7 s** (−26%) | 107.6 → 76.0 s |
| 2 | unchanged (embed-bound) | 16.4 → 10.6 s |

### C — bounded-queue decoupling of parse and store (`61e8a20`)

Parse-result collection no longer awaits storage inline (maxsize-2 queue +
dedicated store consumer; the DB stays a single serial writer), and
streaming mode stops retaining every parsed result in memory.
**Delta: 88.7 → 86.7 s** Tier 1 — modest on 4 cores where parsing itself is
the binding constraint; the structural piece D builds on.

### E — defer cold-index HNSW to end of bulk; unforce per-batch checkpoints (`3b3ea9e`)

Embedding tables created during a bulk run defer their HNSW index to the
final `ensure_all_hnsw_indexes`; embedding batch commits stop forcing
CHECKPOINT (WAL + DuckDB's 16 MB auto-checkpoint provide durability — the
reopen guardrail proves visibility, disproving the old "must checkpoint to
be visible" comment). Ordered before D so overlap would be measured against
a healthy embed phase.

| Metric (Tier 2) | Before | After |
|---|---|---|
| wall | 118.7 s | 100.9 s |
| embed phase | 101.5 s | 83.1 s |
| compaction | 5.5 s | 2.8 s |
| hnsw_ensure | 0.006 s (no-op) | 2.9 s (real, once) |
| **peak DB** | **751 MB** | **101 MB** |

Search snapshot: byte-identical to baseline.

### D — stream embeddings during storage (`4136506`)

Each stored batch's chunks are embedded immediately (provider-recommended
concurrency, one provider request per task, inserts serialized behind a
transaction lock and an async insert wrapper); the missing-embeddings sweep
remains the completeness guarantee and now short-circuits via a COUNT query
instead of loading all chunks twice. Streamed chunks embed **exactly the
text the sweep embeds** so vectors stay identical
(`CHUNKHOUND_STREAM_EMBEDDINGS=0` restores the sweep-only flow).

Full corpus, dims=256, single runs:

| Mode | Wall | Post-storage embed sweep |
|---|---|---|
| sweep-only (E) | 729.7 s | 612.9 s |
| **streamed (D)** | **666.9 s (−8.6%)** | **0.03 s** |

On this box the offline embedder is compute-bound (GIL), so the overlap can
only hide the ~76 s store window; with network-bound providers (OpenAI,
VoyageAI) the overlapped window is *waiting* time and the savings scale with
latency share. A Tier-2 A/B with 150 ms simulated latency showed no
difference for exactly that reason (embed compute ≫ latency at that scale)
— documented rather than cherry-picked.

## Notes and caveats

- The dims=1536 full-corpus rerun after E was progressing with linear file
  growth (0.33 GB where the baseline had already ballooned to many GB) but
  was stopped by the operator at ~90 min: at that scale the *benchmark's own
  offline embedder* is the bottleneck (~17 min of GIL-bound vector math,
  plus DuckDB's Python client converting ~460k floats per batch row by
  row). The dims=256 full-corpus completion above proves the DNF→completes
  claim on an identical failure configuration.
- Real finding for future work: embedding inserts pay a large Python-side
  conversion cost (per-element float conversion in `executemany`); a
  numpy/Arrow registration path would cut embed-phase CPU substantially.
- Pre-existing inconsistency (deliberately not changed): realtime
  (`process_file`) embeds `format_chunk_for_embedding(...)` output (path/
  language headers), while the directory sweep — and now the streamed path,
  to preserve baseline vectors — embeds raw chunk code. Worth an upstream
  decision; whichever format wins, both paths should use it.
- `fake_providers.FakeEmbeddingProvider` gained the protocol-required
  `base_url` property — without it, directory-level embedding with the fake
  provider silently generated 0 embeddings (AttributeError swallowed into
  "generated: 0").

## Correctness evidence

- `tests/test_indexing_pipeline_guardrails.py` (new): fixed-query semantic
  recall (top-1 + top-3 membership), chunks==embeddings invariant,
  incremental semantics (0 files re-processed when unchanged; exactly 1 on
  edit; untouched files keep chunk ids AND embedding rows), exclude
  filtering, close/reopen round-trip. Green at every increment.
- Fixed-query search snapshots byte-identical: baseline vs E, baseline vs
  final (D).
- Full test suite on baseline: 4,251 passed / 119 skipped / 0 failed
  (56 errors were all `tests/site/*` needing `npm ci` + proxy CA —
  environmental, fixed). Final build: **4,345 passed / 108 skipped / 0
  failed** (one test updated: `test_per_file_transaction_isolated` injected
  its fault into a method the batched path legitimately bypasses; it now
  injects at a point shared by both store paths and verifies the same
  isolation contract through batch-rollback → per-file fallback).
- Compaction-test fixture now checkpoints explicitly (embedding inserts no
  longer checkpoint as a side effect).

## Reproduce

```bash
# corpus
pip download Django==5.0 --no-deps --no-binary :all: -d /tmp/corpus
tar -C /tmp/corpus -xzf /tmp/corpus/Django-5.0.tar.gz

# tier 1 (parse+store)
uv run python scripts/bench_indexing.py --corpus /tmp/corpus/Django-5.0 \
    --runs 3 --no-embeddings --json tier1.json

# tier 2 (subset + embeddings + search snapshot)
cp -r /tmp/corpus/Django-5.0/django /tmp/corpus/django-core
rm -rf /tmp/corpus/django-core/contrib
uv run python scripts/bench_indexing.py --corpus /tmp/corpus/django-core \
    --runs 3 --dims 256 --json tier2.json --verify-search snapshot.json

# full pipeline (the baseline-DNF configuration)
uv run python scripts/bench_indexing.py --corpus /tmp/corpus/Django-5.0 \
    --runs 1 --dims 256 --max-db-gb 7 --json full.json
```

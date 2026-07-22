#!/usr/bin/env python3
"""End-to-end indexing benchmark for the ChunkHound pipeline.

Drives the same production code path as `chunkhound index` (configure_registry
-> IndexingCoordinator -> DirectoryIndexingService.process_directory) against a
pinned corpus, using a deterministic offline embedding provider so the numbers
measure the PIPELINE (discovery/parse/store/embed orchestration), not a model
or the network.

Each measured run executes in a fresh subprocess (clean registry, clean RSS
accounting) against a fresh database directory.

Usage:
  # 3 cold-index runs (median reported), embeddings included
  uv run python scripts/bench_indexing.py --corpus /path/to/corpus --runs 3

  # Parse+store only (no embedding phase)
  uv run python scripts/bench_indexing.py --corpus /path --runs 3 --no-embeddings

  # Simulate network latency per embedding batch (milliseconds)
  uv run python scripts/bench_indexing.py --corpus /path --embed-latency-ms 150

  # Dump top-K search results after indexing (for cross-build comparison)
  uv run python scripts/bench_indexing.py --corpus /path --runs 1 --verify-search out.json

Environment knobs honored by the pipeline itself (useful for A/B):
  CHUNKHOUND_DB_BATCH_SIZE, CHUNKHOUND_MP_START_METHOD
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import resource
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent

# Fixed queries used by --verify-search. Chosen to hit distinct subsystems of a
# typical web-framework corpus; adjust per corpus if needed, but keep pinned
# between baseline/after runs.
VERIFY_QUERIES = [
    "database connection pooling and transaction management",
    "http request middleware processing",
    "form field validation errors",
    "template rendering context",
    "user authentication password hashing",
    "cache backend key expiry",
    "migration schema alter table",
    "url routing pattern resolver",
]
VERIFY_TOP_K = 10
VERIFY_REGEX_PATTERNS = [
    r"def get_queryset",
    r"class Middleware",
    r"raise ValidationError",
]


def _eprint(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# Single-run mode: runs inside a fresh subprocess
# ---------------------------------------------------------------------------


def _make_bench_provider(dims: int, latency_ms: float):
    """Create the deterministic offline embedding provider used for benchmarks.

    Subclasses the test-suite FakeEmbeddingProvider (character n-gram hashing,
    deterministic, similar code -> similar vectors) but:
      - computes vectors off the event loop (a real provider awaits network
        I/O, so vector math must not serialize the pipeline through the loop)
      - supports optional simulated per-request latency
      - provides base_url (queried by EmbeddingService token estimation)
    """
    sys.path.insert(0, str(REPO_ROOT))
    from tests.fixtures.fake_providers import FakeEmbeddingProvider

    class BenchEmbeddingProvider(FakeEmbeddingProvider):
        def __init__(self) -> None:
            super().__init__(model="bench-embeddings", dims=dims)
            self._latency_s = latency_ms / 1000.0

        @property
        def base_url(self) -> str | None:
            return None

        async def embed(self, texts: list[str]) -> list[list[float]]:
            if not texts:
                return []
            if self._latency_s > 0:
                await asyncio.sleep(self._latency_s)
            self._requests_made += 1
            self._embeddings_generated += len(texts)

            def _vectors() -> list[list[float]]:
                return [self._generate_deterministic_vector(t) for t in texts]

            return await asyncio.to_thread(_vectors)

    return BenchEmbeddingProvider()


class _PhaseTimer:
    """Wraps coordinator/provider entry points to record wall-clock phases."""

    def __init__(self) -> None:
        self.phases: dict[str, float] = {}
        self.counts: dict[str, int] = {}

    def wrap_async(self, obj: Any, attr: str, phase: str) -> None:
        orig = getattr(obj, attr)

        async def timed(*args: Any, **kwargs: Any) -> Any:
            t0 = time.perf_counter()
            try:
                return await orig(*args, **kwargs)
            finally:
                dt = time.perf_counter() - t0
                self.phases[phase] = self.phases.get(phase, 0.0) + dt
                self.counts[phase] = self.counts.get(phase, 0) + 1

        setattr(obj, attr, timed)

    def wrap_sync(self, obj: Any, attr: str, phase: str) -> None:
        orig = getattr(obj, attr)

        def timed(*args: Any, **kwargs: Any) -> Any:
            t0 = time.perf_counter()
            try:
                return orig(*args, **kwargs)
            finally:
                dt = time.perf_counter() - t0
                self.phases[phase] = self.phases.get(phase, 0.0) + dt
                self.counts[phase] = self.counts.get(phase, 0) + 1

        setattr(obj, attr, timed)


async def _verify_search(db: Any, provider: Any, out_path: str) -> None:
    """Run fixed queries and dump top-K results for cross-build comparison."""
    from chunkhound.services.search_service import SearchService

    service = SearchService(db, provider)
    snapshot: dict[str, Any] = {"semantic": {}, "regex": {}}
    for query in VERIFY_QUERIES:
        results, _ = await service.search_semantic(
            query, page_size=VERIFY_TOP_K, force_strategy="single_hop"
        )
        snapshot["semantic"][query] = [
            {
                "file": r.get("file_path"),
                "symbol": r.get("symbol"),
                "start_line": r.get("start_line"),
                "score": round(float(r.get("similarity", r.get("score", 0.0))), 6),
            }
            for r in results
        ]
    for pattern in VERIFY_REGEX_PATTERNS:
        results, pagination = db.search_regex(pattern=pattern, page_size=VERIFY_TOP_K)
        snapshot["regex"][pattern] = {
            "total": pagination.get("total"),
            "top": [
                {
                    "file": r.get("file_path"),
                    "start_line": r.get("start_line"),
                }
                for r in results
            ],
        }
    Path(out_path).write_text(json.dumps(snapshot, indent=2, sort_keys=True))


async def _single_run_async(args: argparse.Namespace) -> dict[str, Any]:
    corpus = Path(args.corpus).resolve()
    db_dir = Path(args.db_dir).resolve()

    from chunkhound.core.config.config import Config
    from chunkhound.registry import (
        configure_registry,
        create_indexing_coordinator,
        get_registry,
    )
    from chunkhound.services.directory_indexing_service import (
        DirectoryIndexingService,
    )

    config = Config(
        target_dir=corpus,
        database={"path": str(db_dir), "provider": "duckdb"},
    )

    t_connect0 = time.perf_counter()
    configure_registry(config)
    t_connect = time.perf_counter() - t_connect0

    provider = None
    if not args.no_embeddings:
        provider = _make_bench_provider(args.dims, args.embed_latency_ms)
        get_registry().register_provider("embedding", provider, singleton=True)

    coordinator = create_indexing_coordinator()
    db = get_registry().get_provider("database")

    timer = _PhaseTimer()
    timer.wrap_async(coordinator, "_discover_files", "discovery")
    timer.wrap_async(coordinator, "_process_files_in_batches", "parse_store")
    timer.wrap_async(coordinator, "generate_missing_embeddings", "embed")
    timer.wrap_async(coordinator, "compact_database_with_metrics", "compaction")
    timer.wrap_sync(db, "drop_all_hnsw_indexes", "hnsw_drop")
    timer.wrap_sync(db, "ensure_all_hnsw_indexes", "hnsw_ensure")

    service = DirectoryIndexingService(
        indexing_coordinator=coordinator,
        config=config,
    )

    t0 = time.perf_counter()
    stats = await service.process_directory(corpus, no_embeddings=args.no_embeddings)
    wall_s = time.perf_counter() - t0

    final_stats = await coordinator.get_stats()

    if args.verify_search and provider is not None:
        await _verify_search(db, provider, args.verify_search)

    db.close()

    ru_self = resource.getrusage(resource.RUSAGE_SELF)
    ru_children = resource.getrusage(resource.RUSAGE_CHILDREN)

    db_size = 0
    for p in db_dir.rglob("*"):
        if p.is_file():
            db_size += p.stat().st_size

    return {
        "wall_s": round(wall_s, 3),
        "connect_s": round(t_connect, 3),
        "files_processed": stats.files_processed,
        "files_skipped": stats.files_skipped,
        "files_errors": stats.files_errors,
        "chunks_created": stats.chunks_created,
        "embeddings_generated": stats.embeddings_generated,
        "db_files": final_stats.get("files", 0),
        "db_chunks": final_stats.get("chunks", 0),
        "db_embeddings": final_stats.get("embeddings", 0),
        "files_per_s": round(stats.files_processed / wall_s, 2) if wall_s else 0.0,
        "chunks_per_s": round(stats.chunks_created / wall_s, 2) if wall_s else 0.0,
        "peak_rss_self_mb": round(ru_self.ru_maxrss / 1024, 1),
        "peak_rss_child_mb": round(ru_children.ru_maxrss / 1024, 1),
        "db_size_mb": round(db_size / (1024 * 1024), 1),
        "phases_s": {k: round(v, 3) for k, v in sorted(timer.phases.items())},
        "phase_calls": timer.counts,
    }


def _single_run(args: argparse.Namespace) -> None:
    result = asyncio.run(_single_run_async(args))
    print("BENCH_RESULT_JSON:" + json.dumps(result))


# ---------------------------------------------------------------------------
# Driver mode: spawn N isolated runs, report median
# ---------------------------------------------------------------------------


def _drive(args: argparse.Namespace) -> None:
    runs: list[dict[str, Any]] = []
    for i in range(args.runs):
        db_dir = tempfile.mkdtemp(prefix="chunkhound-bench-db-")
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--single-run",
            "--corpus",
            str(args.corpus),
            "--db-dir",
            db_dir,
            "--dims",
            str(args.dims),
            "--embed-latency-ms",
            str(args.embed_latency_ms),
        ]
        if args.no_embeddings:
            cmd.append("--no-embeddings")
        if args.verify_search and i == 0:
            cmd += ["--verify-search", args.verify_search]

        _eprint(f"[bench] run {i + 1}/{args.runs} (db={db_dir})")
        t0 = time.perf_counter()
        proc = subprocess.run(cmd, capture_output=True, text=True)
        dt = time.perf_counter() - t0
        try:
            line = next(
                ln
                for ln in proc.stdout.splitlines()
                if ln.startswith("BENCH_RESULT_JSON:")
            )
        except StopIteration:
            _eprint(f"[bench] run {i + 1} FAILED (rc={proc.returncode}, {dt:.1f}s)")
            _eprint(proc.stdout[-4000:])
            _eprint(proc.stderr[-4000:])
            sys.exit(1)
        result = json.loads(line.split(":", 1)[1])
        _eprint(
            f"[bench] run {i + 1}: wall={result['wall_s']}s "
            f"files={result['files_processed']} chunks={result['chunks_created']} "
            f"embeddings={result['embeddings_generated']} "
            f"rss_self={result['peak_rss_self_mb']}MB phases={result['phases_s']}"
        )
        runs.append(result)
        if not args.keep_dbs:
            shutil.rmtree(db_dir, ignore_errors=True)

    wall = [r["wall_s"] for r in runs]
    median_wall = statistics.median(wall)
    median_run = min(runs, key=lambda r: abs(r["wall_s"] - median_wall))

    summary = {
        "corpus": str(args.corpus),
        "runs": len(runs),
        "no_embeddings": bool(args.no_embeddings),
        "embed_latency_ms": args.embed_latency_ms,
        "dims": args.dims,
        "wall_s_all": wall,
        "wall_s_median": median_wall,
        "median_run": median_run,
        "env": {
            "CHUNKHOUND_DB_BATCH_SIZE": os.environ.get("CHUNKHOUND_DB_BATCH_SIZE"),
            "cpus": os.cpu_count(),
        },
        "all_runs": runs,
    }
    if args.json:
        Path(args.json).write_text(json.dumps(summary, indent=2))
        _eprint(f"[bench] wrote {args.json}")

    print(json.dumps({k: v for k, v in summary.items() if k != "all_runs"}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True, help="Directory to index")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--no-embeddings", action="store_true")
    parser.add_argument("--dims", type=int, default=1536)
    parser.add_argument(
        "--embed-latency-ms",
        type=float,
        default=0.0,
        help="Simulated per-request embedding latency (network stand-in)",
    )
    parser.add_argument("--json", help="Write full summary JSON here")
    parser.add_argument(
        "--verify-search",
        help="After run 1, dump fixed-query top-K results to this JSON path",
    )
    parser.add_argument("--keep-dbs", action="store_true")
    # single-run (internal) mode
    parser.add_argument("--single-run", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--db-dir", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.single_run:
        if not args.db_dir:
            parser.error("--single-run requires --db-dir")
        _single_run(args)
    else:
        _drive(args)


if __name__ == "__main__":
    main()

"""Correctness guardrails for the batch indexing pipeline.

These tests pin the user-visible contracts that indexing-performance work must
not regress:

1. Semantic search recall: fixed queries over a fixture corpus must keep
   returning the expected chunks in the top results (deterministic offline
   embeddings; ordering asserted with tolerance, membership strictly).
2. Every stored chunk gets an embedding (no store/embed pipeline drops).
3. Incremental semantics: an unchanged corpus re-index processes 0 files;
   a single-file edit re-processes exactly that file and preserves the
   embeddings (and chunk ids) of untouched files.
4. Exclude filtering: excluded paths never reach the database.
5. Reopen round-trip: after closing and reopening the database, counts and
   search results survive (no data lost in WAL/checkpoint/compaction).

They exercise the REAL pipeline: DirectoryIndexingService ->
IndexingCoordinator -> DuckDBProvider (file-backed) with the deterministic
FakeEmbeddingProvider from tests.fixtures.
"""

import os
from pathlib import Path

import pytest

from chunkhound.core.config.indexing_config import IndexingConfig
from chunkhound.providers.database.duckdb_provider import DuckDBProvider
from chunkhound.registry import get_registry
from chunkhound.services.directory_indexing_service import DirectoryIndexingService
from chunkhound.services.indexing_coordinator import IndexingCoordinator
from chunkhound.services.search_service import SearchService

from tests.fixtures.fake_providers import FakeEmbeddingProvider

pytestmark = [
    pytest.mark.skipif(
        os.environ.get("CHUNKHOUND_ALLOW_PROCESSPOOL", "0") != "1",
        reason="Requires ProcessPool-friendly environment (SemLock).",
    ),
    pytest.mark.asyncio,
]


class _Config:
    """Minimal config accepted by DirectoryIndexingService/IndexingCoordinator."""

    def __init__(self, **indexing_kwargs) -> None:
        self.indexing = IndexingConfig(**indexing_kwargs)
        self.embedding = None
        self.database = None


# Corpus: files with deliberately distinct vocabulary clusters so the
# deterministic n-gram embeddings produce unambiguous nearest neighbours.
CORPUS: dict[str, str] = {
    "auth.py": '''\
"""User authentication."""


def hash_password(password: str, salt: str) -> str:
    """Hash a user password with the given salt for authentication."""
    return f"{salt}:{password}"


def verify_password(password: str, hashed: str) -> bool:
    """Verify a user password against the stored authentication hash."""
    salt, expected = hashed.split(":", 1)
    return hash_password(password, salt) == hashed
''',
    "database_pool.py": '''\
"""Database connection pooling."""


class ConnectionPool:
    """Pool of database connections with transaction management."""

    def acquire_connection(self):
        """Acquire a pooled database connection and begin a transaction."""
        return object()

    def release_connection(self, conn) -> None:
        """Release the database connection back into the pool after commit."""
        del conn
''',
    "cache_backend.py": '''\
"""Cache backend."""


class CacheBackend:
    """Key-value cache backend with expiry timeouts."""

    def cache_get(self, cache_key: str):
        """Get a cached value by cache key unless the expiry passed."""
        return None

    def cache_set(self, cache_key: str, value, expiry_seconds: int) -> None:
        """Store a value in the cache backend with an expiry timeout."""
        del cache_key, value, expiry_seconds
''',
    "template_render.py": '''\
"""Template rendering."""


def render_template(template_name: str, context: dict) -> str:
    """Render a template with the given context variables."""
    return template_name + str(sorted(context))


def render_to_string(template_name: str, context: dict) -> str:
    """Render a template into a string using the rendering context."""
    return render_template(template_name, context)
''',
    "url_routing.py": '''\
"""URL routing."""


class UrlResolver:
    """Resolve url patterns for request routing."""

    def resolve_route(self, url_path: str):
        """Resolve a url path against registered routing patterns."""
        return url_path


def register_url_pattern(pattern: str) -> None:
    """Register a url pattern with the routing resolver."""
    del pattern
''',
    "docs/overview.md": """\
# Overview

This corpus exists to pin semantic search behaviour for indexing guardrails.

## Search

Queries must keep resolving to the right modules.
""",
}

EXCLUDED_FILES: dict[str, str] = {
    "vendored/skipme.py": "def vendored():\n    return 'must never be indexed'\n",
    "notes.log": "log line one\n",
}

# (query, expected file) — expected file must appear in top-3 semantic results
# and be the top-1 hit. Verified against the baseline build; a pipeline change
# that breaks either has damaged recall (missing chunks/embeddings) and must
# not ship.
RECALL_QUERIES: list[tuple[str, str]] = [
    ("hash a user password for authentication", "auth.py"),
    ("acquire a pooled database connection transaction", "database_pool.py"),
    ("cached value expiry timeout", "cache_backend.py"),
    ("render a template with context variables", "template_render.py"),
    ("resolve url path routing patterns", "url_routing.py"),
]


def _write_corpus(root: Path) -> None:
    for rel, content in {**CORPUS, **EXCLUDED_FILES}.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")


def _build_stack(db_path: Path, root: Path, config: _Config):
    """Create provider + coordinator + service the same way the CLI does."""
    db = DuckDBProvider(db_path, base_directory=root)
    db.connect()
    provider = FakeEmbeddingProvider()
    registry = get_registry()
    registry.register_provider("database", db, singleton=True)
    registry.register_provider("embedding", provider, singleton=True)
    parsers = registry._language_parsers
    coordinator = IndexingCoordinator(
        db, root, provider, parsers, None, config
    )
    service = DirectoryIndexingService(indexing_coordinator=coordinator, config=config)
    return db, provider, coordinator, service


def _config() -> _Config:
    return _Config(exclude=["**/vendored/**", "**/*.log"])


async def _semantic_top(db, provider, query: str, k: int = 3):
    service = SearchService(db, provider)
    results, _ = await service.search_semantic(
        query, page_size=k, force_strategy="single_hop"
    )
    return results


async def test_recall_embeddings_exclusions_and_reopen(tmp_path: Path) -> None:
    root = tmp_path / "corpus"
    root.mkdir()
    _write_corpus(root)
    db_path = tmp_path / "db"

    config = _config()
    db, provider, coordinator, service = _build_stack(db_path, root, config)
    try:
        stats = await service.process_directory(root, no_embeddings=False)
        assert stats.files_processed == len(CORPUS), (
            f"expected {len(CORPUS)} files, processed {stats.files_processed}, "
            f"errors={stats.errors_encountered}"
        )
        assert stats.chunks_created > 0
        assert stats.files_errors == 0

        db_stats = await coordinator.get_stats()
        assert db_stats["files"] == len(CORPUS)
        assert db_stats["chunks"] > 0
        # Contract: every stored chunk got an embedding.
        assert db_stats["embeddings"] == db_stats["chunks"], (
            f"chunks={db_stats['chunks']} but embeddings={db_stats['embeddings']}"
        )

        # Exclude filtering: excluded paths must never be indexed.
        rows = db.execute_query("SELECT path FROM files", [])
        stored_paths = {r["path"] for r in rows}
        assert stored_paths == set(CORPUS), (
            f"stored files diverge from corpus: {sorted(stored_paths)}"
        )
        for excluded in EXCLUDED_FILES:
            assert excluded not in stored_paths

        # Semantic recall on fixed queries.
        for query, expected_file in RECALL_QUERIES:
            results = await _semantic_top(db, provider, query)
            files = [r.get("file_path") for r in results]
            assert expected_file in files, (
                f"query {query!r}: expected {expected_file} in top-3, got {files}"
            )
            assert files[0] == expected_file, (
                f"query {query!r}: expected {expected_file} as top hit, got {files}"
            )

        # Regex search round-trip.
        regex_results, _ = db.search_regex(pattern=r"def hash_password", page_size=5)
        assert any(r.get("file_path") == "auth.py" for r in regex_results)

        baseline_counts = dict(db_stats)
        baseline_top = {
            q: [r.get("file_path") for r in await _semantic_top(db, provider, q)]
            for q, _ in RECALL_QUERIES
        }
    finally:
        db.close()

    # Reopen: data and search behaviour must survive a full close/reopen.
    db2 = DuckDBProvider(db_path, base_directory=root)
    db2.connect()
    try:
        reopened_stats = db2.get_stats()
        for key in ("files", "chunks", "embeddings"):
            assert reopened_stats[key] == baseline_counts[key], (
                f"{key}: reopen={reopened_stats[key]} != baseline={baseline_counts[key]}"
            )
        for query, _ in RECALL_QUERIES:
            top = [r.get("file_path") for r in await _semantic_top(db2, provider, query)]
            assert top == baseline_top[query], (
                f"query {query!r}: reopen top-3 {top} != baseline {baseline_top[query]}"
            )
        regex_results, _ = db2.search_regex(pattern=r"def hash_password", page_size=5)
        assert any(r.get("file_path") == "auth.py" for r in regex_results)
    finally:
        db2.close()


async def test_incremental_reindex_only_changed_files(tmp_path: Path) -> None:
    root = tmp_path / "corpus"
    root.mkdir()
    _write_corpus(root)
    db_path = tmp_path / "db"

    config = _config()
    db, provider, coordinator, service = _build_stack(db_path, root, config)
    try:
        await service.process_directory(root, no_embeddings=False)

        def chunk_map() -> dict[str, list[int]]:
            rows = db.execute_query(
                "SELECT f.path AS path, c.id AS id FROM chunks c "
                "JOIN files f ON c.file_id = f.id ORDER BY c.id",
                [],
            )
            mapping: dict[str, list[int]] = {}
            for r in rows:
                mapping.setdefault(r["path"], []).append(r["id"])
            return mapping

        def embedding_map() -> dict[int, int]:
            rows = db.execute_query(
                "SELECT chunk_id, id FROM embeddings_1536", []
            )
            return {r["chunk_id"]: r["id"] for r in rows}

        chunks_before = chunk_map()
        embeddings_before = embedding_map()

        # 1) No changes -> nothing re-processed, nothing re-embedded.
        stats2 = await service.process_directory(root, no_embeddings=False)
        assert stats2.files_processed == 0, (
            f"unchanged corpus must skip all files, processed {stats2.files_processed}"
        )
        assert stats2.embeddings_generated == 0
        assert chunk_map() == chunks_before

        # 2) Edit one file -> exactly that file re-processed; untouched files
        #    keep their chunk ids AND embedding rows (smart diff contract).
        target = root / "auth.py"
        target.write_text(
            CORPUS["auth.py"]
            + '\n\ndef rotate_password_salt(user_id: int) -> str:\n'
            + '    """Rotate the stored password salt for a user account."""\n'
            + '    return str(user_id)\n',
            encoding="utf-8",
        )
        stats3 = await service.process_directory(root, no_embeddings=False)
        assert stats3.files_processed == 1, (
            f"expected only auth.py re-processed, got {stats3.files_processed}"
        )

        chunks_after = chunk_map()
        embeddings_after = embedding_map()
        for path, ids in chunks_before.items():
            if path == "auth.py":
                continue
            assert chunks_after[path] == ids, (
                f"untouched file {path} chunk ids changed: {ids} -> {chunks_after[path]}"
            )
            for chunk_id in ids:
                assert embeddings_after.get(chunk_id) == embeddings_before[chunk_id], (
                    f"embedding row for untouched chunk {chunk_id} ({path}) changed"
                )

        # New content must be searchable and embedded.
        db_stats = await coordinator.get_stats()
        assert db_stats["embeddings"] == db_stats["chunks"]
        regex_results, _ = db.search_regex(
            pattern=r"def rotate_password_salt", page_size=5
        )
        assert any(r.get("file_path") == "auth.py" for r in regex_results)
    finally:
        db.close()

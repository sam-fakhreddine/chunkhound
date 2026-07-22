from pathlib import Path

from chunkhound.services.indexing_coordinator import IndexingCoordinator
from chunkhound.providers.database.duckdb_provider import DuckDBProvider
from chunkhound.core.types.common import Language
from chunkhound.services.batch_processor import ParsedFileResult


def _pfr(path: Path, chunks: list[dict], ok: bool = True) -> ParsedFileResult:
    return ParsedFileResult(
        file_path=path,
        chunks=chunks if ok else [],
        language=Language.YAML,
        file_size=10,
        file_mtime=0.0,
        status="ok" if ok else "error",
        error=None if ok else "simulated",
        content_hash=None,
    )


def test_per_file_transaction_isolated(tmp_path: Path):
    db = DuckDBProvider(db_path=tmp_path / "db", base_directory=tmp_path)
    db.connect()
    try:
        _run_test(db, tmp_path)
    finally:
        db.disconnect()


def _run_test(db: DuckDBProvider, tmp_path: Path) -> None:
    coord = IndexingCoordinator(database_provider=db, base_directory=tmp_path)

    good_file = tmp_path / "good.yaml"
    bad_file = tmp_path / "bad.yaml"
    good_chunks = [
        {
            "symbol": "a",
            "code": "x: 1",
            "start_line": 1,
            "end_line": 1,
            "chunk_type": "key_value",
            "language": Language.YAML.value,
        }
    ]
    bad_chunks = [
        {
            "symbol": "b",
            "code": "y: 2",
            "start_line": 1,
            "end_line": 1,
            "chunk_type": "key_value",
            "language": Language.YAML.value,
        }
    ]

    # Inject a per-file failure at a point shared by BOTH store paths (the
    # batched path and its per-file fallback): batch storage must roll back
    # and the per-file retry must then isolate the failure to bad_file.
    original_diff = coord._diff_and_validate_chunks

    def _failing_diff(result, file_id, existing_chunks):
        if Path(result.file_path) == bad_file:
            raise RuntimeError("boom")
        return original_diff(result, file_id, existing_chunks)

    coord._diff_and_validate_chunks = _failing_diff  # type: ignore[assignment]

    results = [
        _pfr(good_file, good_chunks, ok=True),
        _pfr(bad_file, bad_chunks, ok=True),
    ]

    import asyncio
    res = asyncio.run(coord._store_parsed_results(results))  # type: ignore[arg-type]
    stats = res[0] if isinstance(res, tuple) else res

    # Good file should be stored; bad file should be in errors
    assert stats["total_files"] == 1
    assert stats["errors"] and any("boom" in e.get("error", "") for e in stats["errors"])  # noqa: SIM115

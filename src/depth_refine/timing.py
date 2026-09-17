"""Per-video wall time records, independent of optional model dependencies."""
from contextlib import contextmanager
from datetime import datetime, timezone
import json
from pathlib import Path
from time import perf_counter


@contextmanager
def episode_timer(result_root: Path, *, stage: str, camera: str,
                  source: Path, **metadata):
    """Persist the latest attempt; callers explicitly mark successful completion."""
    record = dict(stage=stage, camera=camera, chunk=source.parent.parent.name,
                  episode=source.stem, source=str(source), status="skipped",
                  started_at=datetime.now(timezone.utc).isoformat(), **metadata)
    start = perf_counter()
    try:
        yield record
    except BaseException as exc:
        record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        record["elapsed_seconds"] = perf_counter() - start
        path = (result_root / "timings" / stage / record["chunk"] / camera /
                f"{source.stem}.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record, indent=2), encoding="utf-8")
        print(f"[timing] {stage} {record['chunk']}/{source.stem} {camera}: "
              f"{record['elapsed_seconds']:.3f}s ({record['status']})")

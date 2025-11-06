from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from crbot.config import LOG_DIR

LOG_FILE = (LOG_DIR / "matches.jsonl").resolve()
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)


@dataclass
class MatchLog:
    started_at: float
    ended_at: float
    duration_s: float
    crowns_for: int | None = None
    crowns_against: int | None = None


def write_match(start_ts: float, end_ts: float, **extras) -> None:
    record = MatchLog(
        started_at=start_ts,
        ended_at=end_ts,
        duration_s=end_ts - start_ts,
        **extras,
    )
    with LOG_FILE.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")

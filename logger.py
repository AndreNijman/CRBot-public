# logger.py
import json, time, os
from dataclasses import dataclass, asdict

LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "matches.jsonl")
os.makedirs(LOG_DIR, exist_ok=True)

@dataclass
class MatchLog:
    started_at: float
    ended_at: float
    duration_s: float
    crowns_for: int | None = None
    crowns_against: int | None = None
    # add more later: elixir_overflow_time_s, tower_hp, etc.

def write_match(start_ts: float, end_ts: float, **extras):
    rec = MatchLog(
        started_at=start_ts,
        ended_at=end_ts,
        duration_s=end_ts - start_ts,
        **extras
    )
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")

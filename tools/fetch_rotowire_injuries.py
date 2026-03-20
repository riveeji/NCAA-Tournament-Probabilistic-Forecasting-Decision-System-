from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.market_data_utils import DEFAULT_FUZZY_THRESHOLD, attach_team_ids, write_unmatched_log


EXTERNAL_DIR = ROOT / "external-data"
RAW_DIR = EXTERNAL_DIR / "raw-rotowire"
AUDIT_DIR = EXTERNAL_DIR / "audit-logs"
SOURCE_URL = "https://www.rotowire.com/cbasketball/injury-report.php"
JSON_URL = "https://www.rotowire.com/cbasketball/tables/injury-report.php?team=ALL&pos=ALL&conf=ALL&site=other&slateID=None"
HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "X-Requested-With": "XMLHttpRequest",
    "Referer": SOURCE_URL,
}

STATUS_SEVERITY = {
    "Out For Season": 3,
    "Out": 2,
    "Game Time Decision": 1,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch men NCAA injury report from RotoWire into normalized CSV.")
    parser.add_argument("--season", type=int, default=2026, help="Season value to write into output.")
    parser.add_argument("--output-dir", default=str(EXTERNAL_DIR), help="Directory for cleaned output CSV files.")
    parser.add_argument("--raw-dir", default=str(RAW_DIR), help="Directory for raw JSON archives.")
    parser.add_argument(
        "--unmatched-dir",
        default=str(AUDIT_DIR),
        help="Directory for unresolved team-name audit CSVs.",
    )
    parser.add_argument(
        "--fuzzy-threshold",
        type=float,
        default=DEFAULT_FUZZY_THRESHOLD,
        help="Minimum RapidFuzz token_set_ratio score required to auto-map a team name.",
    )
    return parser.parse_args()


def fetch_json() -> list[dict[str, object]]:
    response = requests.get(JSON_URL, headers=HEADERS, timeout=30)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, list):
        raise RuntimeError("Unexpected RotoWire injury response shape")
    return payload


def normalize_rows(payload: list[dict[str, object]], season: int, fuzzy_threshold: float) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    fetched_at = datetime.now(timezone.utc).isoformat()
    rows = []
    for item in payload:
        rows.append(
            {
                "Season": int(season),
                "TeamName": str(item.get("team", "")).strip(),
                "PlayerID": pd.to_numeric(item.get("ID"), errors="coerce"),
                "PlayerName": str(item.get("player", "")).strip(),
                "FirstName": str(item.get("firstname", "")).strip(),
                "LastName": str(item.get("lastname", "")).strip(),
                "Position": str(item.get("position", "")).strip(),
                "Injury": str(item.get("injury", "")).strip(),
                "Status": str(item.get("status", "")).strip(),
                "EstReturnText": str(item.get("rDate", "")).strip(),
                "PlayerURL": str(item.get("playerURL", "")).strip(),
                "Source": "rotowire_injury_report",
                "SourceURL": SOURCE_URL,
                "SnapshotTime": fetched_at,
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame, None
    frame = attach_team_ids(frame, "M", "TeamName", "TeamName", "TeamID", "_TeamIDDiscard", fuzzy_threshold=fuzzy_threshold)
    audit_df = frame.attrs.get("team_match_audit")
    frame["TeamID"] = pd.to_numeric(frame.get("TeamID"), errors="coerce").astype("Int64")
    frame["PlayerID"] = pd.to_numeric(frame.get("PlayerID"), errors="coerce").astype("Int64")
    frame["Severity"] = frame["Status"].map(STATUS_SEVERITY).fillna(0).astype(int)
    frame["IsOut"] = frame["Status"].isin(["Out", "Out For Season"]).astype(int)
    frame["IsGameTimeDecision"] = frame["Status"].eq("Game Time Decision").astype(int)
    frame = frame.drop(columns=["_TeamIDDiscard"], errors="ignore")
    return frame, audit_df


def write_json(path: Path, payload: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    raw_dir = Path(args.raw_dir)
    unmatched_dir = Path(args.unmatched_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    unmatched_dir.mkdir(parents=True, exist_ok=True)

    payload = fetch_json()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    raw_path = raw_dir / f"M_{timestamp}.json"
    output_path = output_dir / f"MRotoWireInjuries_{args.season}.csv"
    unmatched_path = unmatched_dir / f"MRotoWireInjuries_{args.season}_unmatched.csv"

    write_json(raw_path, payload)
    frame, audit_df = normalize_rows(payload, args.season, float(args.fuzzy_threshold))
    frame.to_csv(output_path, index=False)
    write_unmatched_log(audit_df, unmatched_path)

    print(f"[M] source=rotowire_injury_report")
    print(f"[M] raw_rows={len(payload)} cleaned_rows={len(frame)}")
    print(f"[M] saved raw -> {raw_path}")
    print(f"[M] saved csv -> {output_path}")
    if unmatched_path.exists():
        unresolved = pd.read_csv(unmatched_path)
        print(f"[M] unmatched_audit={unmatched_path} rows={len(unresolved)}")


if __name__ == "__main__":
    main()

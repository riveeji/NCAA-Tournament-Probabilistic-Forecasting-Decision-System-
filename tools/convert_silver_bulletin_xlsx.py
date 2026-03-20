from __future__ import annotations

import argparse
import re
import sys
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from zizzii_features import attach_team_ids_from_names

EXTERNAL_DIR = ROOT / "external-data"
NS_MAIN = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
NS_REL = {"r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships"}
NS_PKG = {"pr": "http://schemas.openxmlformats.org/package/2006/relationships"}


def _excel_col_to_index(ref: str) -> int:
    letters = "".join(ch for ch in ref if ch.isalpha())
    value = 0
    for ch in letters:
        value = value * 26 + (ord(ch.upper()) - 64)
    return value - 1


def _shared_strings(zf: zipfile.ZipFile) -> list[str]:
    path = "xl/sharedStrings.xml"
    if path not in zf.namelist():
        return []
    root = ET.fromstring(zf.read(path))
    values: list[str] = []
    for si in root.findall("x:si", NS_MAIN):
        text_parts: list[str] = []
        for node in si.iter():
            if node.tag.endswith("}t") and node.text is not None:
                text_parts.append(node.text)
        values.append("".join(text_parts))
    return values


def _first_sheet_path(zf: zipfile.ZipFile) -> str:
    workbook = ET.fromstring(zf.read("xl/workbook.xml"))
    rels = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
    rel_map = {
        rel.attrib["Id"]: rel.attrib["Target"]
        for rel in rels.findall("pr:Relationship", NS_PKG)
        if "Id" in rel.attrib and "Target" in rel.attrib
    }
    sheet = workbook.find("x:sheets/x:sheet", {**NS_MAIN, **NS_REL})
    if sheet is None:
        raise ValueError("No worksheet found in workbook")
    rel_id = sheet.attrib.get("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id")
    target = rel_map.get(rel_id or "")
    if not target:
        raise ValueError("Worksheet relationship target missing")
    return f"xl/{target.lstrip('/')}"


def read_first_sheet_rows(path: Path) -> list[list[str]]:
    with zipfile.ZipFile(path) as zf:
        strings = _shared_strings(zf)
        sheet_path = _first_sheet_path(zf)
        root = ET.fromstring(zf.read(sheet_path))
    rows: list[list[str]] = []
    for row in root.findall("x:sheetData/x:row", NS_MAIN):
        row_map: dict[int, str] = {}
        max_idx = -1
        for cell in row.findall("x:c", NS_MAIN):
            ref = cell.attrib.get("r", "")
            col_idx = _excel_col_to_index(ref) if ref else max_idx + 1
            max_idx = max(max_idx, col_idx)
            cell_type = cell.attrib.get("t", "")
            if cell_type == "inlineStr":
                node = cell.find("x:is/x:t", NS_MAIN)
                value = node.text if node is not None and node.text is not None else ""
            else:
                value_node = cell.find("x:v", NS_MAIN)
                raw_value = value_node.text if value_node is not None and value_node.text is not None else ""
                if cell_type == "s":
                    try:
                        value = strings[int(raw_value)]
                    except Exception:
                        value = raw_value
                else:
                    value = raw_value
            row_map[col_idx] = value
        rows.append([row_map.get(idx, "") for idx in range(max_idx + 1)])
    return rows


def _normalize_header(value: str) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text


def _find_header_index(rows: list[list[str]], required_headers: set[str]) -> int:
    for idx, row in enumerate(rows):
        headers = {_normalize_header(value) for value in row if str(value).strip()}
        if required_headers.issubset(headers):
            return idx
    raise ValueError(f"Could not find header row containing {sorted(required_headers)}")


def _build_frame(rows: list[list[str]], header_idx: int) -> pd.DataFrame:
    headers = [_normalize_header(value) or f"col_{idx}" for idx, value in enumerate(rows[header_idx])]
    data_rows = rows[header_idx + 1 :]
    width = len(headers)
    shaped = []
    for row in data_rows:
        current = list(row[:width]) + [""] * max(0, width - len(row))
        shaped.append(current[:width])
    frame = pd.DataFrame(shaped, columns=headers)
    frame = frame.replace("", pd.NA)
    frame = frame.dropna(how="all")
    return frame.reset_index(drop=True)


def _parse_snapshot_from_filename(path: Path) -> pd.Timestamp:
    stem = path.stem
    match = re.search(r"(March|April)_(\d{1,2})(?:_(\d{1,2})(\d{2})(am|pm))?", stem, re.IGNORECASE)
    if not match:
        return pd.NaT
    month_name = match.group(1).title()
    day = int(match.group(2))
    if match.group(3) is None:
        hour = 0
        minute = 0
    else:
        hour = int(match.group(3))
        minute = int(match.group(4) or 0)
        meridiem = (match.group(5) or "pm").lower()
        if meridiem == "pm" and hour != 12:
            hour += 12
        if meridiem == "am" and hour == 12:
            hour = 0
    month = 3 if month_name == "March" else 4
    return pd.Timestamp(year=2025, month=month, day=day, hour=hour, minute=minute, tz="UTC")


def convert_team_ratings(path: Path, season: int, gender: str) -> pd.DataFrame:
    rows = read_first_sheet_rows(path)
    header_idx = _find_header_index(rows, {"team", "adjusted_composite", "s_curve"})
    frame = _build_frame(rows, header_idx)
    rename_map = {
        "team": "TeamName",
        "quasi_sagarin": "SB_QuasiSagarin",
        "composite": "SB_Composite",
        "injury_adjustment": "SB_InjuryAdjustment",
        "adjusted_composite": "SB_AdjustedComposite",
        "sbcb_bayesian": "SB_SBCB",
        "pomeroy": "SB_Pomeroy",
        "moore": "SB_Moore",
        "espn_bpi": "SB_ESPNBPI",
        "massey": "SB_Massey",
        "s_curve": "SB_SCurve",
    }
    available = [column for column in rename_map if column in frame.columns]
    frame = frame[available].rename(columns={column: rename_map[column] for column in available})
    frame["Season"] = int(season)
    frame["SnapshotDate"] = _parse_snapshot_from_filename(path)
    frame["Source"] = "Silver Bulletin"
    frame["VerifiedPreTourney"] = 1
    frame = attach_team_ids_from_names(frame, gender, team_col="TeamName", target_col="TeamID")
    numeric_cols = [column for column in frame.columns if column.startswith("SB_")]
    for column in numeric_cols:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    ordered = ["Season", "TeamID", "TeamName", "SnapshotDate", "Source", "VerifiedPreTourney"] + numeric_cols
    return frame[ordered].dropna(subset=["TeamName"]).reset_index(drop=True)


def convert_tournament_forecasts(path: Path, season: int, gender: str) -> pd.DataFrame:
    rows = read_first_sheet_rows(path)
    try:
        header_idx = _find_header_index(rows, {"team_name", "rd1_win", "rd7_win"})
        modern_layout = False
    except ValueError:
        header_idx = _find_header_index(rows, {"team", "seed", "region", "champion"})
        modern_layout = True
    frame = _build_frame(rows, header_idx)
    if modern_layout:
        rename_map = {
            "team": "TeamName",
            "seed": "SB_TeamSeed",
            "region": "SB_TeamRegion",
            "playin": "SB_PlayInFlag",
            "alive": "SB_TeamAlive",
            "rating": "SB_TeamRating",
            "rd64": "SB_Rd1Win",
            "rd32": "SB_Rd2Win",
            "sweet16": "SB_Rd3Win",
            "elite8": "SB_Rd4Win",
            "final4": "SB_Rd5Win",
            "finals": "SB_Rd6Win",
            "champion": "SB_Rd7Win",
        }
    else:
        rename_map = {
            "team_name": "TeamName",
            "team_seed": "SB_TeamSeed",
            "team_region": "SB_TeamRegion",
            "playin_flag": "SB_PlayInFlag",
            "team_alive": "SB_TeamAlive",
            "rd1_win": "SB_Rd1Win",
            "rd2_win": "SB_Rd2Win",
            "rd3_win": "SB_Rd3Win",
            "rd4_win": "SB_Rd4Win",
            "rd5_win": "SB_Rd5Win",
            "rd6_win": "SB_Rd6Win",
            "rd7_win": "SB_Rd7Win",
        }
    raw_timestamp = None
    if modern_layout and "timestamp" in frame.columns:
        raw_timestamp = pd.to_datetime(frame["timestamp"], errors="coerce", utc=True)
    available = [column for column in rename_map if column in frame.columns]
    frame = frame[available].rename(columns={column: rename_map[column] for column in available})
    frame["Season"] = int(season)
    if modern_layout and raw_timestamp is not None:
        timestamps = raw_timestamp.dropna()
        frame["SnapshotDate"] = timestamps.iloc[0] if not timestamps.empty else _parse_snapshot_from_filename(path)
    else:
        frame["SnapshotDate"] = _parse_snapshot_from_filename(path)
    frame["Source"] = "Silver Bulletin"
    frame["VerifiedPreTourney"] = 1
    frame = attach_team_ids_from_names(frame, gender, team_col="TeamName", target_col="TeamID")
    numeric_cols = [
        "SB_TeamSeed",
        "SB_PlayInFlag",
        "SB_TeamAlive",
        "SB_TeamRating",
        "SB_Rd1Win",
        "SB_Rd2Win",
        "SB_Rd3Win",
        "SB_Rd4Win",
        "SB_Rd5Win",
        "SB_Rd6Win",
        "SB_Rd7Win",
    ]
    for column in numeric_cols:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    ordered = [
        "Season",
        "TeamID",
        "TeamName",
        "SnapshotDate",
        "Source",
        "VerifiedPreTourney",
        "SB_TeamSeed",
        "SB_TeamRegion",
        "SB_PlayInFlag",
        "SB_TeamAlive",
        "SB_TeamRating",
        "SB_Rd1Win",
        "SB_Rd2Win",
        "SB_Rd3Win",
        "SB_Rd4Win",
        "SB_Rd5Win",
        "SB_Rd6Win",
        "SB_Rd7Win",
    ]
    available_ordered = [column for column in ordered if column in frame.columns]
    return frame[available_ordered].dropna(subset=["TeamName"]).reset_index(drop=True)


def _default_output_name(kind: str, gender: str, season: int, snapshot: pd.Timestamp) -> str:
    if kind == "ratings":
        return f"{gender}SilverBulletinTeamRatings_{season}.csv"
    suffix = snapshot.strftime("%Y%m%d") if not pd.isna(snapshot) else str(season)
    return f"{gender}SilverBulletinTournamentProbs_{suffix}.csv"


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert Silver Bulletin xlsx files into HC-friendly CSVs.")
    parser.add_argument("--kind", required=True, choices=("ratings", "tourney-probs"))
    parser.add_argument("--gender", required=True, choices=("M", "W"))
    parser.add_argument("--season", type=int, default=2025)
    parser.add_argument("--input", dest="inputs", nargs="+", required=True)
    parser.add_argument("--output", default="", help="Optional output CSV path when converting a single file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    converter = convert_team_ratings if args.kind == "ratings" else convert_tournament_forecasts
    for raw_path in args.inputs:
        input_path = Path(raw_path)
        frame = converter(input_path, season=args.season, gender=args.gender)
        snapshot = pd.to_datetime(frame["SnapshotDate"], errors="coerce").dropna()
        snapshot_value = snapshot.iloc[0] if not snapshot.empty else pd.NaT
        if args.output and len(args.inputs) == 1:
            output_path = Path(args.output)
        else:
            output_path = EXTERNAL_DIR / _default_output_name(args.kind, args.gender, args.season, snapshot_value)
        _write_csv(frame, output_path)
        mapped = int(pd.to_numeric(frame.get("TeamID"), errors="coerce").notna().sum())
        print(
            {
                "input": str(input_path),
                "output": str(output_path),
                "rows": int(len(frame)),
                "mapped_team_ids": mapped,
                "unmapped_team_ids": int(len(frame) - mapped),
            }
        )


if __name__ == "__main__":
    main()

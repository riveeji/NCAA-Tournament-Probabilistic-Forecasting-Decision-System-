from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.goldshot_utils import apply_override_guardrails, favorite_direction_change


TRUTHY = {"1", "true", "yes", "y", "apply"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply automatic and optional manual goldshot overrides to a submission.")
    parser.add_argument("--current", required=True, help="Current HC submission candidate.")
    parser.add_argument("--candidates", required=True, help="CSV from build_goldshot_override_candidates.py")
    parser.add_argument("--manual-shortlist", default="", help="Optional edited manual shortlist CSV.")
    parser.add_argument("--output", required=True, help="Output submission path.")
    parser.add_argument("--summary-output", required=True, help="Summary JSON path.")
    parser.add_argument("--changes-output", default="", help="Optional changed rows CSV path.")
    return parser.parse_args()


def parse_manual_apply(value: object) -> bool:
    text = str(value).strip().lower()
    return text in TRUTHY


def load_submission(path: str) -> pd.DataFrame:
    return pd.read_csv(path, usecols=["ID", "Pred"])


def main() -> None:
    args = parse_args()
    current = load_submission(args.current).rename(columns={"Pred": "CurrentPred"})
    candidates = pd.read_csv(args.candidates)
    candidates = candidates.drop_duplicates(subset=["ID"], keep="first").copy()
    out = current.merge(candidates, on="ID", how="left")
    out["Pred"] = pd.to_numeric(out["CurrentPred"], errors="coerce")

    auto_mask = out["AutoApply"].fillna(False).astype(bool)
    out.loc[auto_mask, "Pred"] = pd.to_numeric(out.loc[auto_mask, "AutoNewPred"], errors="coerce")

    manual_applied_ids: list[str] = []
    manual_source = Path(args.manual_shortlist) if args.manual_shortlist else None
    if manual_source is not None and manual_source.exists():
        manual = pd.read_csv(manual_source)
        if not manual.empty:
            manual = manual.drop_duplicates(subset=["ID"], keep="first").copy()
            manual["ManualApplyBool"] = manual.get("ManualApply", "").map(parse_manual_apply)
            manual["ManualTargetProbNum"] = pd.to_numeric(manual.get("ManualTargetProb", ""), errors="coerce")
            manual = manual.loc[manual["ManualApplyBool"] | manual["ManualTargetProbNum"].notna()].copy()
            if len(manual) > 10:
                manual = manual.head(10).copy()
            manual_lookup = manual.set_index("ID").to_dict("index")
            if manual_lookup:
                for idx, row in out.iterrows():
                    manual_row = manual_lookup.get(str(row["ID"]))
                    if manual_row is None:
                        continue
                    target_prob = manual_row.get("ManualTargetProbNum")
                    if pd.isna(target_prob):
                        target_prob = row.get("TargetProb")
                    if pd.isna(target_prob):
                        continue
                    new_pred = apply_override_guardrails(
                        float(row["CurrentPred"]),
                        float(target_prob),
                        0.50,
                        market_prob=row.get("MarketProbMedian"),
                        spread=row.get("SpreadMedian"),
                        model_prob=row.get("ModelMatchupProbMedian"),
                    )
                    if bool(row.get("FavoriteHasInjuryVeto", False)) and favorite_direction_change(
                        float(row["CurrentPred"]), float(new_pred), str(row.get("FavoriteSide", "Unknown"))
                    ):
                        continue
                    out.at[idx, "Pred"] = float(new_pred)
                    manual_applied_ids.append(str(row["ID"]))

    out["Pred"] = pd.to_numeric(out["Pred"], errors="coerce")
    out["Pred"] = np.clip(out["Pred"].to_numpy(dtype=float), 1e-6, 1.0 - 1e-6)
    submission = out[["ID", "Pred"]].copy().sort_values("ID").reset_index(drop=True)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.output, index=False)

    changed = out.loc[(pd.to_numeric(out["Pred"], errors="coerce") - pd.to_numeric(out["CurrentPred"], errors="coerce")).abs() > 1e-12].copy()
    changed["AppliedMode"] = np.where(
        changed["ID"].astype(str).isin(manual_applied_ids),
        "manual",
        np.where(changed["AutoApply"].fillna(False).astype(bool), "auto", "other"),
    )
    if args.changes_output:
        Path(args.changes_output).parent.mkdir(parents=True, exist_ok=True)
        changed.to_csv(args.changes_output, index=False)

    summary = {
        "current": str(Path(args.current).resolve()),
        "candidates": str(Path(args.candidates).resolve()),
        "manual_shortlist": str(manual_source.resolve()) if manual_source is not None and manual_source.exists() else "",
        "output": str(Path(args.output).resolve()),
        "rows": int(len(submission)),
        "auto_applied_rows": int((changed["AppliedMode"] == "auto").sum()),
        "manual_applied_rows": int((changed["AppliedMode"] == "manual").sum()),
        "total_changed_rows": int(len(changed)),
        "auto_applied_playin_rows": int(((changed["AppliedMode"] == "auto") & changed.get("IsPlayIn", False).fillna(False)).sum()) if not changed.empty and "IsPlayIn" in changed.columns else 0,
        "auto_applied_round1_rows": int(((changed["AppliedMode"] == "auto") & changed.get("IsRound1", False).fillna(False)).sum()) if not changed.empty and "IsRound1" in changed.columns else 0,
        "auto_applied_round2_rows": int(((changed["AppliedMode"] == "auto") & changed.get("IsRound2", False).fillna(False)).sum()) if not changed.empty and "IsRound2" in changed.columns else 0,
        "auto_model_supported_rows": int(((changed["AppliedMode"] == "auto") & changed.get("AutoModelSupported", False).fillna(False)).sum()) if not changed.empty and "AutoModelSupported" in changed.columns else 0,
        "max_abs_change": float((pd.to_numeric(changed["Pred"], errors="coerce") - pd.to_numeric(changed["CurrentPred"], errors="coerce")).abs().max()) if not changed.empty else 0.0,
        "changed_ids_top10": changed[["ID", "CurrentPred", "Pred", "AppliedMode"]].head(10).where(pd.notna(changed), None).to_dict("records"),
        "changes_output": str(Path(args.changes_output).resolve()) if args.changes_output else "",
    }
    Path(args.summary_output).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"goldshot submission written to: {args.output}")
    print(f"goldshot summary written to: {args.summary_output}")
    if args.changes_output:
        print(f"goldshot changes written to: {args.changes_output}")


if __name__ == "__main__":
    main()

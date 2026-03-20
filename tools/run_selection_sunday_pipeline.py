from __future__ import annotations

import argparse
import os
import traceback
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = ROOT / "tools"
DEFAULT_BOOKMAKERS = "fanduel,draftkings,betmgm,caesars,betrivers,bet365"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the March Machine Learning Mania 2026 Selection Sunday refresh pipeline."
    )
    parser.add_argument(
        "--api-key",
        default="",
        help="The Odds API key. Defaults to THE_ODDS_API_KEY env var.",
    )
    parser.add_argument(
        "--skip-check",
        action="store_true",
        help="Skip official Kaggle data completeness check.",
    )
    parser.add_argument(
        "--skip-ratings",
        action="store_true",
        help="Skip public ratings refresh.",
    )
    parser.add_argument(
        "--skip-odds",
        action="store_true",
        help="Skip live odds refresh.",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip final model retraining.",
    )
    parser.add_argument(
        "--skip-hc",
        action="store_true",
        help="Skip HC submission generation.",
    )
    parser.add_argument(
        "--skip-final-check",
        action="store_true",
        help="Skip final single-submission hash and probability-distribution validation.",
    )
    parser.add_argument(
        "--skip-candidate-reports",
        action="store_true",
        help="Skip generating HC/baseline candidate comparison reports.",
    )
    parser.add_argument(
        "--gender",
        choices=["M", "W", "all"],
        default="all",
        help="Fetch odds for one side or both.",
    )
    parser.add_argument(
        "--final-submission",
        default="submission_stage2_single_final_hc.csv",
        help="Final chosen Stage 2 submission file to validate.",
    )
    parser.add_argument(
        "--baseline-submission",
        default="submission_stage2_single_final.csv",
        help="Baseline candidate submission file.",
    )
    parser.add_argument(
        "--regions",
        default="us",
        help="The Odds API region list for current odds fetches.",
    )
    parser.add_argument(
        "--bookmakers",
        default=DEFAULT_BOOKMAKERS,
        help="Preferred bookmaker keys for the first The Odds API pass.",
    )
    return parser.parse_args()


def run_step(label: str, command: list[str], env: dict[str, str] | None = None) -> None:
    print()
    print(f"[step] {label}")
    print(" ".join(command))
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def run_optional_step(label: str, command: list[str], env: dict[str, str] | None = None) -> None:
    try:
        run_step(label, command, env=env)
    except subprocess.CalledProcessError:
        print()
        print(f"[warn] optional step failed: {label}")
        print(traceback.format_exc())


def submission_matches_sample(submission_path: Path, sample_path: Path) -> bool:
    if not submission_path.exists() or not sample_path.exists():
        return False
    try:
        sub = pd.read_csv(submission_path, usecols=["ID", "Pred"])
        sample = pd.read_csv(sample_path, usecols=["ID"])
    except Exception:
        return False
    if len(sub) != len(sample):
        return False
    if sub["ID"].duplicated().any() or sub["Pred"].isna().any():
        return False
    return set(sub["ID"]) == set(sample["ID"])


def market_file_metrics(path: Path) -> dict[str, float]:
    if not path.exists():
        return {
            "rows": 0.0,
            "spread_coverage": 0.0,
            "unique_matchups": 0.0,
            "marketprob_matchups": 0.0,
        }
    try:
        frame = pd.read_csv(path)
    except Exception:
        return {
            "rows": 0.0,
            "spread_coverage": 0.0,
            "unique_matchups": 0.0,
            "marketprob_matchups": 0.0,
        }
    if frame.empty:
        return {
            "rows": 0.0,
            "spread_coverage": 0.0,
            "unique_matchups": 0.0,
            "marketprob_matchups": 0.0,
        }
    if "LastSpread" in frame.columns:
        spread_cov = float(pd.to_numeric(frame["LastSpread"], errors="coerce").notna().mean())
    else:
        spread_cov = 0.0
    unique_matchups = 0.0
    marketprob_matchups = 0.0
    if {"T1", "T2"}.issubset(frame.columns):
        grouped = frame.groupby(["T1", "T2"], dropna=False)
        unique_matchups = float(grouped.ngroups)
        if "MarketProb" in frame.columns:
            marketprob_matchups = float(
                grouped["MarketProb"].apply(lambda values: pd.to_numeric(values, errors="coerce").notna().any()).sum()
            )
    return {
        "rows": float(len(frame)),
        "spread_coverage": spread_cov,
        "unique_matchups": unique_matchups,
        "marketprob_matchups": marketprob_matchups,
    }


def latest_result_path(pattern: str) -> str:
    matches = sorted((ROOT / "results").glob(pattern), key=lambda path: path.stat().st_mtime, reverse=True)
    return str(matches[0]) if matches else ""


def main() -> None:
    args = parse_args()
    python = sys.executable
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    sample_submission_path = ROOT / "ncaa-data" / "SampleSubmissionStage2.csv"
    baseline_submission_path = (ROOT / args.baseline_submission).resolve()
    silver_summary_json = ""
    goldshot_summary_json = ""
    recommendation_json = ""

    if not args.skip_check:
        run_step(
            "check official Kaggle data",
            [python, str(TOOLS_DIR / "check_competition_data.py")],
        )

    if not args.skip_ratings:
        run_step(
            "refresh public external ratings",
            [python, str(TOOLS_DIR / "fetch_public_external_ratings.py")],
        )
        run_step(
            "refresh CollegeHoopsHub ratings",
            [python, str(TOOLS_DIR / "fetch_collegehoopshub_ratings.py")],
        )

    if not args.skip_odds:
        env = os.environ.copy()
        today_local = datetime.now().date()
        api_key = args.api_key.strip() or env.get("THE_ODDS_API_KEY", "").strip()
        if not api_key:
            print()
            print("[warn] THE_ODDS_API_KEY is missing. Skipping odds refresh.")
        else:
            env["THE_ODDS_API_KEY"] = api_key
            odds_command = [
                python,
                str(TOOLS_DIR / "fetch_theoddsapi_odds.py"),
                "--season",
                "2026",
                "--regions",
                args.regions,
                "--markets",
                "h2h,spreads",
                "--keep-by-book",
            ]
            if args.bookmakers.strip():
                odds_command.extend(["--bookmakers", args.bookmakers.strip()])
            if args.gender == "all":
                odds_command.append("--all")
            else:
                odds_command.extend(["--gender", args.gender])
            run_step("refresh The Odds API matchup odds", odds_command, env=env)

        if api_key and args.gender in {"W", "all"}:
            women_metrics = market_file_metrics(ROOT / "external-data" / "WMatchupOdds_2026.csv")
            women_marketprob_ratio = (
                women_metrics["marketprob_matchups"] / women_metrics["unique_matchups"]
                if women_metrics["unique_matchups"] > 0
                else 0.0
            )
            if (
                women_metrics["rows"] < 4
                or women_metrics["spread_coverage"] <= 0.0
                or women_metrics["marketprob_matchups"] < 24
                or women_marketprob_ratio < 0.85
            ):
                fallback_command = [
                    python,
                    str(TOOLS_DIR / "fetch_theoddsapi_odds.py"),
                    "--gender",
                    "W",
                    "--season",
                    "2026",
                    "--regions",
                    args.regions,
                    "--markets",
                    "h2h,spreads",
                    "--keep-by-book",
                ]
                run_step(
                    "refresh The Odds API women matchup odds without bookmaker restriction",
                    fallback_command,
                    env=env,
                )

        kalshi_command = [
            python,
            str(TOOLS_DIR / "fetch_kalshi_prediction_markets.py"),
            "--season",
            "2026",
            "--date-from",
            today_local.isoformat(),
            "--date-to",
            (today_local + timedelta(days=7)).isoformat(),
        ]
        if args.gender == "all":
            kalshi_command.append("--all")
        else:
            kalshi_command.extend(["--gender", args.gender])
        run_optional_step("refresh Kalshi prediction-market odds", kalshi_command)

        polymarket_command = [
            python,
            str(TOOLS_DIR / "fetch_polymarket_prediction_markets.py"),
            "--season",
            "2026",
            "--date-from",
            today_local.isoformat(),
            "--date-to",
            (today_local + timedelta(days=7)).isoformat(),
        ]
        if args.gender == "all":
            polymarket_command.append("--all")
        else:
            polymarket_command.extend(["--gender", args.gender])
        run_optional_step("refresh Polymarket prediction-market odds", polymarket_command)

        actionnetwork_command = [python, str(TOOLS_DIR / "fetch_actionnetwork_odds.py"), "--season", "2026"]
        if args.gender == "all":
            actionnetwork_command.append("--all")
        else:
            actionnetwork_command.extend(["--gender", args.gender])
        run_optional_step("refresh Action Network current odds supplement", actionnetwork_command)
        warren_command = [
            python,
            str(TOOLS_DIR / "fetch_warrennolan_predict_winners.py"),
            "--season",
            "2026",
            "--date-from",
            today_local.isoformat(),
            "--date-to",
            (today_local + timedelta(days=7)).isoformat(),
            "--type1",
            "NCAA",
            "--type2",
            "All Games",
        ]
        if args.gender == "all":
            warren_command.append("--all")
        else:
            warren_command.extend(["--gender", args.gender])
        run_optional_step("refresh Warren Nolan matchup projections", warren_command)

        barttorvik_command = [
            python,
            str(TOOLS_DIR / "fetch_barttorvik_matchup_projections.py"),
            "--season",
            "2026",
            "--date-from",
            today_local.isoformat(),
            "--date-to",
            (today_local + timedelta(days=7)).isoformat(),
        ]
        if args.gender == "all":
            barttorvik_command.append("--all")
        else:
            barttorvik_command.extend(["--gender", args.gender])
        run_optional_step("refresh Bart Torvik matchup projections", barttorvik_command)

        hhs_email = env.get("HERHOOPSTATS_EMAIL", "").strip()
        hhs_password = env.get("HERHOOPSTATS_PASSWORD", "").strip()
        if hhs_email and hhs_password and args.gender in {"W", "all"}:
            run_optional_step(
                "refresh Her Hoop Stats women matchup projections",
                [
                    python,
                    str(TOOLS_DIR / "fetch_herhoopstats_matchup_projections.py"),
                    "--season",
                    "2026",
                    "--date-from",
                    today_local.isoformat(),
                    "--date-to",
                    (today_local + timedelta(days=7)).isoformat(),
                ],
                env=env,
            )

        if args.gender in {"M", "all"}:
            men_metrics = market_file_metrics(ROOT / "external-data" / "MMatchupOdds_2026.csv")
            if men_metrics["spread_coverage"] < 0.80:
                run_step(
                    "refresh TeamRankings spread-derived men odds",
                    [python, str(TOOLS_DIR / "fetch_teamrankings_odds.py"), "--season", "2026"],
                )

        run_optional_step(
            "refresh men injury report from RotoWire",
            [python, str(TOOLS_DIR / "fetch_rotowire_injuries.py"), "--season", "2026"],
        )
        run_optional_step(
            "build men tournament availability watchlist",
            [python, str(TOOLS_DIR / "build_availability_watchlist.py"), "--season", "2026"],
        )

    if not args.skip_train:
        try:
            run_step(
                "retrain models and regenerate submissions",
                [python, str(ROOT / "zizzii_train.py")],
            )
        except subprocess.CalledProcessError:
            if submission_matches_sample(baseline_submission_path, sample_submission_path):
                print()
                print(
                    "[warn] baseline retrain failed; continuing with the existing baseline candidate "
                    f"at {baseline_submission_path}"
                )
                print(traceback.format_exc())
            else:
                raise

    if not args.skip_hc:
        run_step(
            "generate HC clean-train no-runtime candidate",
            [
                python,
                str(ROOT / "hc" / "predict.py"),
                "--season",
                "2026",
                "--runtime-rules",
                "off",
                "--output",
                "results\\submission_stage2_single_final_hc_no_runtime.csv",
            ],
        )
        run_step(
            "generate HC clean-train silver-runtime candidate",
            [
                python,
                str(ROOT / "hc" / "predict.py"),
                "--season",
                "2026",
                "--runtime-rules",
                "silver",
                "--output",
                "results\\submission_stage2_single_final_hc_silver_runtime.csv",
            ],
        )
        silver_summary_json = latest_result_path("hc_submission_summary_*_submission_stage2_single_final_hc_silver_runtime*.json")
        goldshot_candidates_csv = str(ROOT / "results" / "final_market_override_candidates_2026.csv")
        goldshot_candidates_json = str(ROOT / "results" / f"final_market_override_candidates_{run_id}.json")
        goldshot_manual_csv = str(ROOT / "results" / f"final_manual_override_shortlist_{run_id}.csv")
        goldshot_manual_json = str(ROOT / "results" / f"final_manual_override_shortlist_{run_id}.json")
        goldshot_output = str(ROOT / "results" / "submission_stage2_single_final_hc_goldshot.csv")
        goldshot_changes = str(ROOT / "results" / f"submission_stage2_single_final_hc_goldshot_changes_{run_id}.csv")
        goldshot_summary_json = str(ROOT / "results" / f"submission_stage2_single_final_hc_goldshot_summary_{run_id}.json")
        run_step(
            "build goldshot override candidates",
            [
                python,
                str(TOOLS_DIR / "build_goldshot_override_candidates.py"),
                "--current",
                "results\\submission_stage2_single_final_hc_no_runtime.csv",
                "--baseline",
                args.baseline_submission,
                "--season",
                "2026",
                "--csv-output",
                goldshot_candidates_csv,
                "--json-output",
                goldshot_candidates_json,
                "--manual-csv-output",
                goldshot_manual_csv,
                "--manual-json-output",
                goldshot_manual_json,
            ],
        )
        run_step(
            "apply goldshot overrides",
            [
                python,
                str(TOOLS_DIR / "apply_goldshot_overrides.py"),
                "--current",
                "results\\submission_stage2_single_final_hc_no_runtime.csv",
                "--candidates",
                goldshot_candidates_csv,
                "--manual-shortlist",
                goldshot_manual_csv,
                "--output",
                goldshot_output,
                "--summary-output",
                goldshot_summary_json,
                "--changes-output",
                goldshot_changes,
            ],
        )

    if not args.skip_candidate_reports:
        run_step(
            "compare HC no-runtime vs HC silver-runtime candidates",
            [
                python,
                str(TOOLS_DIR / "build_candidate_diff_report.py"),
                "--candidate-a",
                "results\\submission_stage2_single_final_hc_no_runtime.csv",
                "--candidate-b",
                "results\\submission_stage2_single_final_hc_silver_runtime.csv",
                "--label-a",
                "hc_no_runtime",
                "--label-b",
                "hc_silver_runtime",
                "--summary-output",
                str(ROOT / "results" / f"candidate_diff_hc_no_runtime_vs_silver_{run_id}.json"),
                "--details-output",
                str(ROOT / "results" / f"candidate_diff_hc_no_runtime_vs_silver_{run_id}.csv"),
            ],
        )
        run_step(
            "compare HC no-runtime vs HC goldshot candidates",
            [
                python,
                str(TOOLS_DIR / "build_candidate_diff_report.py"),
                "--candidate-a",
                "results\\submission_stage2_single_final_hc_no_runtime.csv",
                "--candidate-b",
                "results\\submission_stage2_single_final_hc_goldshot.csv",
                "--label-a",
                "hc_current",
                "--label-b",
                "hc_goldshot",
                "--summary-output",
                str(ROOT / "results" / f"candidate_diff_hc_current_vs_goldshot_{run_id}.json"),
                "--details-output",
                str(ROOT / "results" / f"candidate_diff_hc_current_vs_goldshot_{run_id}.csv"),
            ],
        )
        run_step(
            "compare baseline vs HC silver-runtime candidates",
            [
                python,
                str(TOOLS_DIR / "build_candidate_diff_report.py"),
                "--candidate-a",
                args.baseline_submission,
                "--candidate-b",
                "results\\submission_stage2_single_final_hc_silver_runtime.csv",
                "--label-a",
                "baseline",
                "--label-b",
                "hc_silver_runtime",
                "--summary-output",
                str(ROOT / "results" / f"candidate_diff_baseline_vs_hc_silver_{run_id}.json"),
                "--details-output",
                str(ROOT / "results" / f"candidate_diff_baseline_vs_hc_silver_{run_id}.csv"),
            ],
        )
        run_step(
            "compare baseline vs HC goldshot candidates",
            [
                python,
                str(TOOLS_DIR / "build_candidate_diff_report.py"),
                "--candidate-a",
                args.baseline_submission,
                "--candidate-b",
                "results\\submission_stage2_single_final_hc_goldshot.csv",
                "--label-a",
                "baseline",
                "--label-b",
                "hc_goldshot",
                "--summary-output",
                str(ROOT / "results" / f"candidate_diff_baseline_vs_hc_goldshot_{run_id}.json"),
                "--details-output",
                str(ROOT / "results" / f"candidate_diff_baseline_vs_hc_goldshot_{run_id}.csv"),
            ],
        )
        run_step(
            "compare baseline vs HC no-runtime candidates",
            [
                python,
                str(TOOLS_DIR / "build_candidate_diff_report.py"),
                "--candidate-a",
                args.baseline_submission,
                "--candidate-b",
                "results\\submission_stage2_single_final_hc_no_runtime.csv",
                "--label-a",
                "baseline",
                "--label-b",
                "hc_no_runtime",
                "--summary-output",
                str(ROOT / "results" / f"candidate_diff_baseline_vs_hc_no_runtime_{run_id}.json"),
                "--details-output",
                str(ROOT / "results" / f"candidate_diff_baseline_vs_hc_no_runtime_{run_id}.csv"),
            ],
        )
        current_review_csv = str(ROOT / "results" / f"final_review_checklist_hc_current_{run_id}.csv")
        current_review_json = str(ROOT / "results" / f"final_review_checklist_hc_current_{run_id}.json")
        current_shortlist_csv = str(ROOT / "results" / f"final_review_shortlist_hc_current_{run_id}.csv")
        current_shortlist_json = str(ROOT / "results" / f"final_review_shortlist_hc_current_{run_id}.json")
        run_step(
            "build final review checklist for current HC",
            [
                python,
                str(TOOLS_DIR / "build_final_review_checklist.py"),
                "--baseline",
                args.baseline_submission,
                "--hc",
                "results\\submission_stage2_single_final_hc_no_runtime.csv",
                "--season",
                "2026",
                "--csv-output",
                current_review_csv,
                "--json-output",
                current_review_json,
                "--shortlist-csv-output",
                current_shortlist_csv,
                "--shortlist-json-output",
                current_shortlist_json,
            ],
        )
        run_step(
            "build final review checklist for HC goldshot",
            [
                python,
                str(TOOLS_DIR / "build_final_review_checklist.py"),
                "--baseline",
                args.baseline_submission,
                "--hc",
                "results\\submission_stage2_single_final_hc_goldshot.csv",
                "--season",
                "2026",
                "--csv-output",
                str(ROOT / "results" / f"final_review_checklist_hc_goldshot_{run_id}.csv"),
                "--json-output",
                str(ROOT / "results" / f"final_review_checklist_hc_goldshot_{run_id}.json"),
                "--shortlist-csv-output",
                str(ROOT / "results" / f"final_review_shortlist_hc_goldshot_{run_id}.csv"),
                "--shortlist-json-output",
                str(ROOT / "results" / f"final_review_shortlist_hc_goldshot_{run_id}.json"),
            ],
        )
        recommendation_json = str(ROOT / "results" / f"final_submission_recommendation_{run_id}.json")
        run_step(
            "build final submission recommendation",
            [
                python,
                str(TOOLS_DIR / "build_final_submission_recommendation.py"),
                "--current",
                "results\\submission_stage2_single_final_hc_no_runtime.csv",
                "--silver",
                "results\\submission_stage2_single_final_hc_silver_runtime.csv",
                "--goldshot",
                "results\\submission_stage2_single_final_hc_goldshot.csv",
                "--baseline",
                args.baseline_submission,
                "--silver-summary-json",
                silver_summary_json,
                "--goldshot-summary-json",
                goldshot_summary_json,
                "--current-review-csv",
                current_review_csv,
                "--output",
                recommendation_json,
            ],
        )

    if not args.skip_final_check:
        final_submission_target = args.final_submission
        if args.final_submission == "submission_stage2_single_final_hc.csv" and recommendation_json and Path(recommendation_json).exists():
            try:
                recommendation = json.loads(Path(recommendation_json).read_text(encoding="utf-8"))
                recommended_path = str(recommendation.get("recommended_submission", "")).strip()
                if recommended_path and Path(recommended_path).exists():
                    final_submission_target = recommended_path
                    print()
                    print(f"[info] final sanity check will validate the recommended candidate: {final_submission_target}")
            except Exception:
                pass
        summary_output = ROOT / "results" / f"final_submission_check_{run_id}.json"
        run_step(
            "final single-submission hash and probability-distribution check",
            [
                python,
                str(TOOLS_DIR / "check_submission_sanity.py"),
                "--submission",
                final_submission_target,
                "--summary-output",
                str(summary_output),
            ],
        )

    print()
    print("[done] Selection Sunday pipeline finished.")


if __name__ == "__main__":
    main()

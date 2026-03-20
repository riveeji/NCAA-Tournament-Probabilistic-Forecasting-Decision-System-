from __future__ import annotations

import argparse
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


NOTEBOOK_TEMPLATE = """# Kaggle notebook helper for historical HC submission
from pathlib import Path
import shutil
import pandas as pd

INPUT_DIR = Path("/kaggle/input/{dataset_slug}")
src = INPUT_DIR / "{submission_name}"
dst = Path("/kaggle/working/submission.csv")

if not src.exists():
    raise FileNotFoundError(f"Missing bundled submission file: {{src}}")

shutil.copy(src, dst)
df = pd.read_csv(dst)
print(df.head())
print({{"rows": len(df), "nan_pred": int(df["Pred"].isna().sum()), "dup_id": int(df["ID"].duplicated().sum())}})
"""


NOTEBOOK_ALIGN_TEMPLATE = """# Kaggle notebook helper that aligns a bundled submission to the competition sample schema
from pathlib import Path
import pandas as pd

BUNDLE_DIR = Path("/kaggle/input/{dataset_slug}")
COMP_DIR = Path("/kaggle/input/{competition_slug}")

pred_path = BUNDLE_DIR / "{submission_name}"
if not pred_path.exists():
    raise FileNotFoundError(f"Missing bundled submission file: {{pred_path}}")

sample_candidates = list(COMP_DIR.glob("*sample*submission*.csv")) + list(COMP_DIR.glob("*Sample*Submission*.csv")) + list(COMP_DIR.glob("*.csv"))
sample_path = None
for candidate in sample_candidates:
    if "sample" in candidate.name.lower() and "submission" in candidate.name.lower():
        sample_path = candidate
        break
if sample_path is None:
    raise FileNotFoundError(f"Could not find the competition sample submission under {{COMP_DIR}}")

sample = pd.read_csv(sample_path)
pred = pd.read_csv(pred_path)

sample_cols = sample.columns.tolist()
if len(sample_cols) < 2:
    raise ValueError(f"Unexpected sample submission columns: {{sample_cols}}")

key_col = sample_cols[0]
target_col = sample_cols[1]

if "ID" in pred.columns and key_col not in pred.columns:
    pred = pred.rename(columns={{"ID": key_col}})
elif "RowId" in pred.columns and key_col not in pred.columns:
    pred = pred.rename(columns={{"RowId": key_col}})

if "Pred" not in pred.columns:
    raise ValueError(f"Bundled submission must contain a 'Pred' column, got {{pred.columns.tolist()}}")
if key_col not in pred.columns:
    raise ValueError(f"Bundled submission does not contain the required key column '{{key_col}}'")

submission = sample[[key_col]].merge(pred[[key_col, "Pred"]], on=key_col, how="left")
if submission["Pred"].isna().any():
    missing = int(submission["Pred"].isna().sum())
    raise ValueError(f"Missing predictions for {{missing}} sample rows after alignment")

submission = submission.rename(columns={{"Pred": target_col}})
submission = submission[sample_cols]
out_path = Path("/kaggle/working/submission.csv")
submission.to_csv(out_path, index=False)

print(f"Competition sample: {{sample_path}}")
print(submission.head())
print({{"rows": len(submission), "key_col": key_col, "target_col": target_col, "nan_target": int(submission[target_col].isna().sum())}})
"""


NOTEBOOK_MAP_FROM_TEST_TEMPLATE = """# Kaggle notebook helper that maps bundled all-pairs predictions onto the official competition test rows
from pathlib import Path
import pandas as pd

BUNDLE_DIR = Path("/kaggle/input/{dataset_slug}")
COMP_DIR = Path("/kaggle/input/{competition_slug}")

pred_path = BUNDLE_DIR / "{submission_name}"
if not pred_path.exists():
    raise FileNotFoundError(f"Missing bundled submission file: {{pred_path}}")


def pick_sample_submission(comp_dir: Path) -> Path:
    candidates = list(comp_dir.rglob("*.csv"))
    ranked = []
    for path in candidates:
        name = path.name.lower()
        if "sample" in name and "submission" in name:
            ranked.append((0, len(str(path)), path))
        else:
            ranked.append((1, len(str(path)), path))
    ranked.sort()
    for _, _, path in ranked:
        name = path.name.lower()
        if "sample" in name and "submission" in name:
            return path
    raise FileNotFoundError(f"Could not find the competition sample submission under {{comp_dir}}")


def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    return out


def find_test_frame(comp_dir: Path, key_col: str) -> tuple[Path, pd.DataFrame, str, str, str]:
    candidates = []
    for path in comp_dir.rglob("*.csv"):
        name = path.name.lower()
        if "sample" in name and "submission" in name:
            continue
        try:
            df = normalize_cols(pd.read_csv(path))
        except Exception:
            continue
        cols_lower = {{c.lower(): c for c in df.columns}}
        season_col = cols_lower.get("season")
        t1_col = (
            cols_lower.get("t1")
            or cols_lower.get("t1teamid")
            or cols_lower.get("teamid1")
            or cols_lower.get("team1")
            or cols_lower.get("team1id")
        )
        t2_col = (
            cols_lower.get("t2")
            or cols_lower.get("t2teamid")
            or cols_lower.get("teamid2")
            or cols_lower.get("team2")
            or cols_lower.get("team2id")
        )
        if key_col in df.columns and season_col and t1_col and t2_col:
            candidates.append((0 if "test" in name else 1, len(df), len(str(path)), path, df, season_col, t1_col, t2_col))
    if not candidates:
        raise FileNotFoundError(
            "Could not find a competition test CSV containing the sample key column plus Season/T1/T2 columns. "
            "This usually means the competition uses opaque RowId values and requires row-native notebook inference."
        )
    candidates.sort()
    _, _, _, path, df, season_col, t1_col, t2_col = candidates[0]
    return path, df, season_col, t1_col, t2_col


sample_path = pick_sample_submission(COMP_DIR)
sample = normalize_cols(pd.read_csv(sample_path))
pred = normalize_cols(pd.read_csv(pred_path))

sample_cols = sample.columns.tolist()
if len(sample_cols) < 2:
    raise ValueError(f"Unexpected sample submission columns: {{sample_cols}}")

key_col = sample_cols[0]
target_col = sample_cols[1]

if "ID" not in pred.columns:
    raise ValueError(f"Bundled submission must contain an ID column, got {{pred.columns.tolist()}}")
if "Pred" not in pred.columns:
    raise ValueError(f"Bundled submission must contain a Pred column, got {{pred.columns.tolist()}}")

test_path, test_df, season_col, t1_col, t2_col = find_test_frame(COMP_DIR, key_col)

test_df = test_df[[key_col, season_col, t1_col, t2_col]].copy()
test_df[key_col] = test_df[key_col].astype(str).str.strip()
test_df["Season"] = pd.to_numeric(test_df[season_col], errors="coerce").astype("Int64")
test_df["T1"] = pd.to_numeric(test_df[t1_col], errors="coerce").astype("Int64")
test_df["T2"] = pd.to_numeric(test_df[t2_col], errors="coerce").astype("Int64")

if test_df[["Season", "T1", "T2"]].isna().any().any():
    raise ValueError(f"Competition test file {{test_path}} contains non-numeric Season/T1/T2 values")

pred_parts = pred["ID"].astype(str).str.split("_", expand=True)
if pred_parts.shape[1] != 3:
    raise ValueError("Bundled submission ID is not in Season_T1_T2 format")
pred_map = pred.copy()
pred_map["Season"] = pd.to_numeric(pred_parts[0], errors="coerce").astype("Int64")
pred_map["T1"] = pd.to_numeric(pred_parts[1], errors="coerce").astype("Int64")
pred_map["T2"] = pd.to_numeric(pred_parts[2], errors="coerce").astype("Int64")
pred_map = pred_map[["Season", "T1", "T2", "Pred"]].drop_duplicates()

submission = sample[[key_col]].copy()
submission[key_col] = submission[key_col].astype(str).str.strip()
submission = submission.merge(test_df[[key_col, "Season", "T1", "T2"]], on=key_col, how="left")
submission = submission.merge(pred_map, on=["Season", "T1", "T2"], how="left")

missing = int(submission["Pred"].isna().sum())
if missing > 0:
    missing_ids = submission.loc[submission["Pred"].isna(), key_col].head(10).tolist()
    raise ValueError(
        f"Missing predictions for {{missing}} official test rows after Season/T1/T2 mapping. "
        f"Examples: {{missing_ids}}. If the official RowId is opaque without matchup columns, a precomputed all-pairs CSV cannot be used directly."
    )

submission = submission[[key_col, "Pred"]].rename(columns={{"Pred": target_col}})
submission = submission[sample_cols]
out_path = Path("/kaggle/working/submission.csv")
submission.to_csv(out_path, index=False)

print(f"Competition sample: {{sample_path}}")
print(f"Competition test: {{test_path}}")
print(submission.head())
print({{
    "rows": len(submission),
    "key_col": key_col,
    "target_col": target_col,
    "nan_target": int(submission[target_col].isna().sum()),
    "dup_key": int(submission[key_col].duplicated().sum()),
    "saved_to": str(out_path),
}})
"""


README_TEMPLATE = """# Kaggle Notebook Bundle

This bundle is for the easiest notebook-style submission flow.

## Files
- `{submission_name}`: precomputed submission file
- `notebook_submit_copy.py`: code cell template to copy the bundled file to `/kaggle/working/submission.csv`
- `notebook_submit_align_to_sample.py`: code cell template to align the bundled file to the official competition sample schema
- `notebook_submit_map_from_test.py`: code cell template to map all-pairs predictions onto official competition test rows using `Season/T1/T2`

## How to use on Kaggle
1. Create a private Kaggle Dataset from the contents of this folder.
2. Add that dataset to your notebook.
3. If the competition accepts the bundled CSV as-is, use `notebook_submit_copy.py`.
4. If the competition expects a different key column such as `RowId`, use `notebook_submit_align_to_sample.py`.
5. If the competition sample uses opaque row keys but the official test file exposes `Season/T1/T2`, use `notebook_submit_map_from_test.py`.
6. If `notebook_submit_map_from_test.py` still cannot map rows, then the competition requires row-native inference and a precomputed all-pairs CSV is not enough.
7. Run the notebook.
8. Kaggle will pick up `/kaggle/working/submission.csv` as the submission artifact.

Dataset slug placeholder used in the template:
- `{dataset_slug}`

If you rename the dataset, edit the slug in both notebook templates.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a minimal Kaggle notebook bundle around a precomputed submission CSV.")
    parser.add_argument("--submission-file", required=True, help="Path to the generated submission CSV.")
    parser.add_argument("--output-dir", required=True, help="Directory to write the bundle into.")
    parser.add_argument("--dataset-slug", required=True, help="Kaggle dataset slug to reference in the notebook template.")
    args = parser.parse_args()

    submission_path = (ROOT / args.submission_file).resolve() if not Path(args.submission_file).is_absolute() else Path(args.submission_file)
    output_dir = (ROOT / args.output_dir).resolve() if not Path(args.output_dir).is_absolute() else Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not submission_path.exists():
        raise FileNotFoundError(f"Submission file not found: {submission_path}")

    target_submission = output_dir / submission_path.name
    shutil.copy2(submission_path, target_submission)

    (output_dir / "notebook_submit_copy.py").write_text(
        NOTEBOOK_TEMPLATE.format(
            dataset_slug=args.dataset_slug,
            submission_name=submission_path.name,
        ),
        encoding="utf-8",
    )
    (output_dir / "notebook_submit_align_to_sample.py").write_text(
        NOTEBOOK_ALIGN_TEMPLATE.format(
            dataset_slug=args.dataset_slug,
            competition_slug="your-competition-slug",
            submission_name=submission_path.name,
        ),
        encoding="utf-8",
    )
    (output_dir / "notebook_submit_map_from_test.py").write_text(
        NOTEBOOK_MAP_FROM_TEST_TEMPLATE.format(
            dataset_slug=args.dataset_slug,
            competition_slug="your-competition-slug",
            submission_name=submission_path.name,
        ),
        encoding="utf-8",
    )
    (output_dir / "README.md").write_text(
        README_TEMPLATE.format(
            submission_name=submission_path.name,
            dataset_slug=args.dataset_slug,
        ),
        encoding="utf-8",
    )

    print(f"Notebook bundle written to: {output_dir}")
    print(f"Submission copied to: {target_submission}")


if __name__ == "__main__":
    main()

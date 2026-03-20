from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
ASSETS = ROOT / "docs" / "assets"


def ensure_assets() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)


def svg_header(width: int, height: int) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" fill="none">'
    )


def build_architecture_svg() -> None:
    width, height = 1200, 380
    box_w, box_h = 190, 96
    y = 150
    xs = [30, 260, 490, 720, 950]
    labels = [
        ("Data", "Odds, markets, ratings,\nmanual supplements"),
        ("Features", "Team-level and matchup-level\nsignal standardization"),
        ("Models", "Elo, efficiency, boosting,\nmeta fusion"),
        ("Decision", "Runtime rules,\ngoldshot, recommendation"),
        ("Release", "Submission build,\nsanity, hash, reports"),
    ]
    colors = ["#103b45", "#165a72", "#1d7c8d", "#2b9aa0", "#69b578"]
    parts = [svg_header(width, height)]
    parts.append('<rect width="1200" height="380" fill="#f7fafc"/>')
    parts.append(
        '<text x="30" y="48" fill="#0f172a" font-size="28" font-family="Arial, sans-serif" '
        'font-weight="700">System Architecture</text>'
    )
    parts.append(
        '<text x="30" y="78" fill="#475569" font-size="16" font-family="Arial, sans-serif">'
        "End-to-end NCAA tournament forecasting and submission workflow</text>"
    )
    for idx, x in enumerate(xs):
        parts.append(
            f'<rect x="{x}" y="{y}" width="{box_w}" height="{box_h}" rx="18" fill="{colors[idx]}"/>'
        )
        parts.append(
            f'<text x="{x + 18}" y="{y + 34}" fill="#ffffff" font-size="22" '
            'font-family="Arial, sans-serif" font-weight="700">'
            f"{labels[idx][0]}</text>"
        )
        body_lines = labels[idx][1].split("\n")
        body_y = y + 60
        for line in body_lines:
            parts.append(
                f'<text x="{x + 18}" y="{body_y}" fill="#dbeafe" font-size="15" '
                'font-family="Arial, sans-serif">'
                f"{line}</text>"
            )
            body_y += 19
        if idx < len(xs) - 1:
            arrow_x1 = x + box_w
            arrow_x2 = xs[idx + 1] - 14
            mid_y = y + box_h / 2
            parts.append(
                f'<line x1="{arrow_x1 + 10}" y1="{mid_y}" x2="{arrow_x2}" y2="{mid_y}" '
                'stroke="#94a3b8" stroke-width="5" stroke-linecap="round"/>'
            )
            parts.append(
                f'<polygon points="{arrow_x2}, {mid_y} {arrow_x2 - 16},{mid_y - 10} '
                f'{arrow_x2 - 16},{mid_y + 10}" fill="#94a3b8"/>'
            )
    parts.append("</svg>")
    (ASSETS / "system-architecture.svg").write_text("".join(parts), encoding="utf-8")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def to_float(value: str) -> float:
    return float(value)


def scale_points(values: list[float], left: int, top: int, width: int, height: int) -> list[tuple[float, float]]:
    n = len(values)
    min_v = min(values)
    max_v = max(values)
    if max_v == min_v:
        max_v += 1e-6
    points = []
    for idx, val in enumerate(values):
        x = left + (width * idx / (n - 1 if n > 1 else 1))
        y = top + height - ((val - min_v) / (max_v - min_v) * height)
        points.append((x, y))
    return points


def polyline(points: list[tuple[float, float]], color: str) -> str:
    coords = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    return (
        f'<polyline points="{coords}" fill="none" stroke="{color}" '
        'stroke-width="4" stroke-linecap="round" stroke-linejoin="round"/>'
    )


def circles(points: list[tuple[float, float]], color: str) -> str:
    return "".join(
        f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.5" fill="{color}" stroke="#ffffff" stroke-width="2"/>'
        for x, y in points
    )


def build_cv_trend_svg() -> None:
    rows = read_csv(RESULTS / "hc_cv_brier_2003_2025.csv")
    seasons = [row["Season"] for row in rows]
    men = [to_float(row["MenBrier"]) for row in rows]
    women = [to_float(row["WomenBrier"]) for row in rows]
    combined = [to_float(row["EqualGenderMean"]) for row in rows]
    width, height = 1200, 520
    left, top, chart_w, chart_h = 90, 120, 1030, 320
    palette = {
        "Men": "#1d4ed8",
        "Women": "#db2777",
        "Equal-Gender Mean": "#15803d",
    }
    all_vals = men + women + combined
    min_v, max_v = min(all_vals), max(all_vals)
    if max_v == min_v:
        max_v += 1e-6
    parts = [svg_header(width, height)]
    parts.append('<rect width="1200" height="520" fill="#f8fafc"/>')
    parts.append(
        '<text x="30" y="48" fill="#0f172a" font-size="28" font-family="Arial, sans-serif" '
        'font-weight="700">Historical CV Brier Trend</text>'
    )
    parts.append(
        '<text x="30" y="80" fill="#475569" font-size="16" font-family="Arial, sans-serif">'
        "Lower is better. Tracks men, women, and equal-gender mean across seasons.</text>"
    )
    parts.append(f'<rect x="{left}" y="{top}" width="{chart_w}" height="{chart_h}" rx="14" fill="#ffffff" stroke="#e2e8f0"/>')
    for tick in range(5):
        y = top + chart_h * tick / 4
        val = max_v - (max_v - min_v) * tick / 4
        parts.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{left + chart_w}" y2="{y:.1f}" stroke="#e2e8f0" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="24" y="{y + 5:.1f}" fill="#64748b" font-size="14" font-family="Arial, sans-serif">'
            f"{val:.3f}</text>"
        )
    for idx, season in enumerate(seasons):
        x = left + chart_w * idx / (len(seasons) - 1 if len(seasons) > 1 else 1)
        parts.append(
            f'<text x="{x:.1f}" y="{top + chart_h + 28}" text-anchor="middle" fill="#64748b" '
            'font-size="12" font-family="Arial, sans-serif">'
            f"{season}</text>"
        )
    men_pts = scale_points(men, left, top, chart_w, chart_h)
    women_pts = scale_points(women, left, top, chart_w, chart_h)
    combined_pts = scale_points(combined, left, top, chart_w, chart_h)
    for name, pts in [
        ("Men", men_pts),
        ("Women", women_pts),
        ("Equal-Gender Mean", combined_pts),
    ]:
        parts.append(polyline(pts, palette[name]))
        parts.append(circles(pts, palette[name]))
    legend_x = 760
    legend_y = 54
    for idx, name in enumerate(["Men", "Women", "Equal-Gender Mean"]):
        y = legend_y + idx * 24
        parts.append(
            f'<line x1="{legend_x}" y1="{y}" x2="{legend_x + 28}" y2="{y}" stroke="{palette[name]}" stroke-width="5" stroke-linecap="round"/>'
        )
        parts.append(
            f'<text x="{legend_x + 38}" y="{y + 5}" fill="#0f172a" font-size="15" font-family="Arial, sans-serif">{name}</text>'
        )
    parts.append("</svg>")
    (ASSETS / "cv-trend.svg").write_text("".join(parts), encoding="utf-8")


def build_upset_scenarios_svg() -> None:
    rows = read_csv(RESULTS / "scenario_brier_upset_bands_latest.csv")
    width, height = 1200, 500
    parts = [svg_header(width, height)]
    parts.append('<rect width="1200" height="500" fill="#f8fafc"/>')
    parts.append(
        '<text x="30" y="48" fill="#0f172a" font-size="28" font-family="Arial, sans-serif" '
        'font-weight="700">Scenario Brier Under Different Upset Regimes</text>'
    )
    parts.append(
        '<text x="30" y="80" fill="#475569" font-size="16" font-family="Arial, sans-serif">'
        "Lower is better. Combined Brier under light, moderate, and high-chaos tournament worlds.</text>"
    )
    chart_left, chart_top, chart_w, chart_h = 120, 120, 980, 300
    parts.append(f'<rect x="{chart_left}" y="{chart_top}" width="{chart_w}" height="{chart_h}" rx="14" fill="#ffffff" stroke="#e2e8f0"/>')
    vals = [to_float(row["CombinedMeanBrier"]) for row in rows]
    max_v = max(vals) * 1.15
    colors = ["#22c55e", "#f59e0b", "#ef4444"]
    bar_w = 180
    gap = 120
    for idx, row in enumerate(rows):
        val = to_float(row["CombinedMeanBrier"])
        x = chart_left + 110 + idx * (bar_w + gap)
        y = chart_top + chart_h - (val / max_v * (chart_h - 40))
        h = chart_top + chart_h - y
        parts.append(
            f'<rect x="{x}" y="{y:.1f}" width="{bar_w}" height="{h:.1f}" rx="18" fill="{colors[idx]}"/>'
        )
        parts.append(
            f'<text x="{x + bar_w / 2:.1f}" y="{y - 12:.1f}" text-anchor="middle" fill="#0f172a" '
            'font-size="20" font-family="Arial, sans-serif" font-weight="700">'
            f"{val:.3f}</text>"
        )
        parts.append(
            f'<text x="{x + bar_w / 2:.1f}" y="{chart_top + chart_h + 34}" text-anchor="middle" '
            'fill="#0f172a" font-size="16" font-family="Arial, sans-serif" font-weight="700">'
            f"{row['Scenario']}</text>"
        )
        parts.append(
            f'<text x="{x + bar_w / 2:.1f}" y="{chart_top + chart_h + 58}" text-anchor="middle" '
            'fill="#64748b" font-size="13" font-family="Arial, sans-serif">'
            f"Fav win rate: {float(row['FavoriteWinRateMean']):.1%}</text>"
        )
        parts.append(
            f'<text x="{x + bar_w / 2:.1f}" y="{chart_top + chart_h + 78}" text-anchor="middle" '
            'fill="#64748b" font-size="13" font-family="Arial, sans-serif">'
            f"P10-P90: {float(row['CombinedP10']):.3f} - {float(row['CombinedP90']):.3f}</text>"
        )
    parts.append("</svg>")
    (ASSETS / "upset-scenarios.svg").write_text("".join(parts), encoding="utf-8")


def main() -> None:
    ensure_assets()
    build_architecture_svg()
    build_cv_trend_svg()
    build_upset_scenarios_svg()
    print("Generated README figures in", ASSETS)


if __name__ == "__main__":
    main()

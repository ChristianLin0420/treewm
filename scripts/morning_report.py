"""Assemble the morning report, grouped by research axis (spec section 18).

Reads every artifact the night produced and reports per axis: best configuration,
quantitative result, whether it is robust or preliminary, what changed in the tree, and a
promote / drop / combine recommendation. Deliberately not a single-winner summary.

    python scripts/morning_report.py > MORNING_REPORT.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]

AXES = {
    "A": ("recursive training", ["A0_random", "A0_bfs", "A1_multistep", "A2_ss25", "A2_ss50",
                                 "A4_noresidual", "H2_depthw"]),
    "B": ("branch proposal", ["A0_random", "B1_k8", "B2_k16", "B3_k8_short"]),
    "C": ("temporal horizon", ["A0_random", "C1_short", "C2_medium", "C4_fixed8"]),
    "G": ("capacity / training", ["A0_random", "G1_wide", "G2_deep", "G3_z256"]),
    "H": ("loss design", ["A0_random", "H1_bind", "H2_depthw", "H3_noredundancy"]),
}

SCREEN_COLS = [
    ("eval/success_rate", "success", True),
    ("eval/goal_distance_final", "goal_dist", False),
    ("val/loss_total", "val_loss", False),
    ("tree/effective_branching_factor", "EBF", True),
    ("eval/selected_leaf_depth", "leaf_depth", True),
]


def load(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def fmt(v, nd=3):
    return "n/a" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.{nd}f}"


def section_screening(out: list[str]) -> dict:
    data = load(REPO / "results_screen/screen_summary.json")
    if not data:
        out.append("_Phase-1 screening artifact missing._\n")
        return {}
    runs = data["runs"]
    base = runs.get(data["baseline"], {})

    for axis, (title, recipes) in AXES.items():
        out.append(f"\n### Axis {axis} — {title}\n")
        out.append("| recipe | " + " | ".join(c[1] for c in SCREEN_COLS) + " |")
        out.append("|---" * (len(SCREEN_COLS) + 1) + "|")
        best, best_s = None, -1e9
        for r in recipes:
            key = f"{r}_s0"
            if key not in runs:
                continue
            row = runs[key]
            cells = [fmt(row.get(t)) for t, _, _ in SCREEN_COLS]
            mark = " _(baseline)_" if key == data["baseline"] else ""
            out.append(f"| `{r}`{mark} | " + " | ".join(cells) + " |")
            s = row.get("eval/success_rate", -1e9)
            if key != data["baseline"] and s > best_s:
                best, best_s = r, s
        if best:
            b = runs[f"{best}_s0"]
            delta = b.get("eval/success_rate", 0) - base.get("eval/success_rate", 0)
            out.append(f"\n**Best on this axis:** `{best}` "
                       f"(success {fmt(b.get('eval/success_rate'))}, "
                       f"{delta:+.3f} vs baseline; 1 seed -> **preliminary**).")
    return data


def section_phase2(out: list[str]) -> None:
    curves = load(REPO / "results_phase2/budget_curves.json")
    if not curves:
        out.append("\n_Phase-2 budget curves missing._\n")
        return
    budgets = [16, 32, 64, 128, 256]
    by_ds: dict[str, dict[str, list]] = {}
    for key, pts in curves.items():
        ds, arm = key.split("|")
        by_ds.setdefault(ds, {})[arm] = pts
    for ds, arms in by_ds.items():
        out.append(f"\n### Phase 2 — success vs node budget, {ds} (3 seeds)\n")
        out.append("| recipe | " + " | ".join(str(b) for b in budgets) + " | AUC |")
        out.append("|---" * (len(budgets) + 2) + "|")
        rows = []
        for arm, pts in arms.items():
            vals = [float(np.mean(pts[str(b)])) for b in budgets if str(b) in pts]
            if not vals:
                continue
            rows.append((arm, vals, float(np.mean(vals))))
        for arm, vals, auc in sorted(rows, key=lambda r: -r[2]):
            out.append(f"| `{arm}` | " + " | ".join(fmt(v) for v in vals) + f" | **{fmt(auc)}** |")


def section_coverage(out: list[str]) -> None:
    cov = load(REPO / "results_phase2/coverage_curves.json")
    if not cov:
        return
    out.append("\n### Phase 2 — coverage vs node budget (distinct regions)\n")
    for ds, arms in cov.items():
        budgets = sorted({int(b) for a in arms.values() for b in a})
        out.append(f"\n**{ds}**\n")
        out.append("| recipe | " + " | ".join(str(b) for b in budgets) + " |")
        out.append("|---" * (len(budgets) + 1) + "|")
        for arm, pts in arms.items():
            vals = [float(np.mean(pts[str(b)])) if str(b) in pts else None for b in budgets]
            out.append(f"| `{arm}` | " + " | ".join(fmt(v, 1) for v in vals) + " |")


def section_tracks(out: list[str]) -> None:
    data = load(REPO / "results_tracks/tracks_def_pointmaze-medium-stitch.json")
    if not data:
        out.append("\n_Tracks D/E/F artifact missing._\n")
        return
    for track, title in (("D", "tree structure / allocation"),
                         ("E", "execution cadence"),
                         ("F", "planner interface")):
        keys = sorted(k for k in data if k.startswith(f"{track}|"))
        if not keys:
            continue
        out.append(f"\n### Axis {track} — {title}\n")
        cols = sorted({c for k in keys for c in data[k]}, key=lambda c: (not c.isdigit(), c))
        out.append("| setting | " + " | ".join(cols) + " |")
        out.append("|---" * (len(cols) + 1) + "|")
        for k in keys:
            cells = [fmt(float(np.mean(data[k][c]))) if c in data[k] else "n/a" for c in cols]
            out.append(f"| `{k.split('|')[1]}` | " + " | ".join(cells) + " |")
    shapes = data.get("_shapes", {})
    if shapes:
        out.append("\n**Tree shape by policy (budget 64):**\n")
        keys = ["tree/mean_depth", "tree/unique_root_subtrees_explored",
                "tree/top2_root_subtree_fraction"]
        out.append("| policy | " + " | ".join(k.split('/')[-1] for k in keys) + " |")
        out.append("|---" * (len(keys) + 1) + "|")
        for pol, s in shapes.items():
            out.append(f"| `{pol}` | " + " | ".join(fmt(s.get(k)) for k in keys) + " |")


def section_antmaze(out: list[str]) -> None:
    curves = load(REPO / "results_antmaze/budget_curves.json")
    out.append("\n### Axis: generalisation — AntMaze\n")
    if not curves:
        out.append("_AntMaze results missing (pipeline did not reach this stage)._\n")
        return
    for key, pts in curves.items():
        ds, arm = key.split("|")
        vals = {b: float(np.mean(v)) for b, v in pts.items()}
        out.append(f"- `{arm}` on {ds}: " + ", ".join(f"b{b}={fmt(v)}" for b, v in sorted(
            vals.items(), key=lambda kv: int(kv[0]))))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=None)
    args = p.parse_args()

    out: list[str] = ["# TreeWM overnight design-space search — morning report\n"]
    out.append("Screening = 1 seed (**preliminary**); Phase 2 = 3 seeds (**more robust**). "
               "RandomTreeWM (`A0_random`) is the control in every comparison.\n")
    out.append("\n## Phase 1 — mechanism screening (1 seed, pointmaze-medium-stitch)\n")
    section_screening(out)
    out.append("\n## Phase 2 — promoted recipes (3 seeds)\n")
    section_phase2(out)
    section_coverage(out)
    out.append("\n## Tracks D / E / F — no retraining\n")
    section_tracks(out)
    section_antmaze(out)

    text = "\n".join(out)
    if args.out:
        Path(args.out).write_text(text)
        print(f"wrote {args.out}")
    else:
        print(text)


if __name__ == "__main__":
    main()

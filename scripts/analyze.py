"""Aggregate the study's per-run JSON files into per-budget tables.

Reads whatever `experiment.py` has written so far, so it is safe to run on a
partial grid.

For each (budget, algorithm) it collects one number per seed -- the best loss
that seed reached *within that budget* -- and summarises across seeds.

Getting that number right is the only subtle part:

* run was done at this budget            -> use its best_loss
* budget-independent, run at a larger    -> running minimum over the first
  budget (Adam, NA-Adam-fixed)              `budget` evaluations. Free; marked
                                            "(derived)".
* budget-dependent, different budget     -> skipped. VeLO's and NA-Adam-
  (VeLO, NA-Adam-relative)                  relative's whole trajectory is
                                            shaped by the budget, so a run at
                                            another budget says nothing here.

`experiment.py --verify-prefix` tests the derivation rather than trusting it.

Usage:
    python scripts/analyze.py
    python scripts/analyze.py --out-dir results/study --csv results/summary.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

BUDGETS = (250, 500, 1000, 2000, 4000) # = max_evals = num_steps

#: A run whose best loss exceeds this is treated as diverged rather than merely
#: bad. Calibrated against observations on Voyager: healthy runs land in the
#: 20-110 range, while runs that overran VeLO's horizon collapsed to ~385.
DIVERGED_ABOVE = 300.0


def load(out_dir: Path) -> list[dict]:
    records = [json.loads(p.read_text()) for p in sorted(out_dir.glob("*.json"))]
    if not records:
        raise SystemExit(f"no records found in {out_dir}")
    return records


def best_at_budget(record: dict, budget: int) -> float | None:
    """Best loss this run achieved within `budget` evaluations, or None."""
    if record["budget"] == budget:
        return record["best_loss"]
    if not record.get("budget_independent"):
        return None
    if budget > record["budget"]:
        return None
    history = np.asarray(record["loss_history"], dtype=float)[:budget]
    history = history[np.isfinite(history)]
    return float(history.min()) if history.size else None


def summarize(values: list[float]) -> dict:
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    return {
        "n": int(arr.size),
        "median": float(np.median(finite)) if finite.size else math.nan,
        "min": float(finite.min()) if finite.size else math.nan,
        "max": float(finite.max()) if finite.size else math.nan,
        # Diverged/non-finite seeds. A median alone would hide these, and on
        # this problem they are the difference between "works" and "unusable".
        "bad": int((~np.isfinite(arr)).sum() + (finite > DIVERGED_ABOVE).sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out-dir", type=Path, default=Path("results/study"))
    parser.add_argument("--csv", type=Path, default=None)
    args = parser.parse_args()

    # Only score runs from the study stage -- hyperparameter-selection runs
    # must never be reported as results. This reads each record's own `stage`
    # field; anything else (a seed list duplicated from experiment.py, say)
    # goes stale the moment that file is edited.
    all_records = load(args.out_dir)
    records = [r for r in all_records if r.get("stage") == "study"]
    other = len(all_records) - len(records)

    if not records:
        raise SystemExit(
            f"no study records in {args.out_dir} ({len(all_records)} file(s) "
            "found, none with stage='study'). Tuning runs are not scored; run "
            "--stage study first. Records written before the 'stage' field "
            "existed are ignored -- delete or re-run them."
        )
    if other:
        print(f"note: skipped {other} non-study record(s) (tuning, or written "
              "before the 'stage' field existed)")

    table: dict[tuple[int, str], list[float]] = defaultdict(list)
    derived: set[tuple[int, str]] = set()

    for budget in BUDGETS:
        for record in records:
            value = best_at_budget(record, budget)
            if value is None:
                continue
            key = (budget, record["label"])
            table[key].append(value)
            if record["budget"] != budget:
                derived.add(key)

    print(
        f"\nProblem: {records[0]['problem']}   "
        f"seeds: {sorted({r['seed'] for r in records})}   "
        f"(diverged = best loss > {DIVERGED_ABOVE:g} or non-finite)\n"
    )

    rows = []
    for budget in BUDGETS:
        scored = [
            (summarize(values)["median"], label, summarize(values))
            for (b, label), values in table.items()
            if b == budget
        ]
        if not scored:
            continue

        print(f"=== budget {budget} ===")
        print(f"{'algorithm':<46}{'n':>4}{'median':>11}{'min':>11}"
              f"{'max':>11}{'bad':>5}")
        for median, label, stats in sorted(scored):
            tag = " (derived)" if (budget, label) in derived else ""
            print(
                f"{label + tag:<46}{stats['n']:>4}{stats['median']:>11.4f}"
                f"{stats['min']:>11.4f}{stats['max']:>11.4f}{stats['bad']:>5}"
            )
            rows.append(
                {"budget": budget, "algorithm": label,
                 "derived": (budget, label) in derived, **stats}
            )

        # Headline: does the learned optimizer beat the best hand-designed one?
        velo = next((s for _, l, s in scored if l.startswith("velo")), None)
        baselines = [(m, l) for m, l, _ in scored if not l.startswith("velo")]
        if velo and baselines and math.isfinite(velo["median"]):
            best_median, best_label = min(baselines)
            verdict = "VeLO better" if velo["median"] < best_median else "baseline better"
            ratio = velo["median"] / best_median if best_median > 0 else math.nan
            print(
                f"  -> {verdict}: velo {velo['median']:.4f} vs "
                f"{best_label} {best_median:.4f}  (ratio {ratio:.2f}x)"
            )
        print()

    if args.csv and rows:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {args.csv}")


if __name__ == "__main__":
    main()

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

def load(out_dir: Path) -> list[dict]:
    records = [json.loads(p.read_text()) for p in sorted(out_dir.glob("*.json"))]
    if not records:
        raise SystemExit(f"no records found in {out_dir}")
    return records


def best_at_budget(record: dict, budget: int) -> tuple[float, int, float] | None:
    """(best loss, step it occurred at, last loss) within `budget` evals.

    Returns None when this run says nothing about this budget.

    Both the step and the last loss are recomputed for derived budgets:
    truncating a run to its first N evaluations moves both the argmin and the
    endpoint, so the record's own `best_at` / `final_loss` are only valid at
    the budget the run was actually made with.
    """
    if record["budget"] == budget:
        return record["best_loss"], record["best_at"], record["final_loss"]
    if not record.get("budget_independent"):
        return None
    if budget > record["budget"]:
        return None
    history = np.asarray(record["loss_history"], dtype=float)[:budget]
    if not np.any(np.isfinite(history)):
        return None
    step = int(np.nanargmin(history))
    return float(history[step]), step, float(history[-1])


def summarize(entries: list[tuple[float, int, float]]) -> dict:
    """Aggregate across seeds.

    `min` is the best result any seed reached -- i.e. what you get from
    restarting this many times and keeping the best. `min_at` and `final`
    describe *that same seed's* run: the step its best came at, and the loss it
    ended on. The gap between `min` and `final` is how far it drifted after its
    best point; `min_at` near the budget with `final` close to `min` means it
    was still converging at the end.

    `max` is the worst seed, which is where a diverged run shows up. Note `min`
    improves with more seeds, so only compare it between algorithms at equal
    `n`.

    Non-finite results are counted rather than silently dropped; ignoring them
    would overstate an algorithm that sometimes blows up.
    """
    values = np.asarray([v for v, _, _ in entries], dtype=float)
    steps = [s for _, s, _ in entries]
    finals = [f for _, _, f in entries]
    finite = np.isfinite(values)

    if not finite.any():
        return {"n": values.size, "min": math.nan, "min_at": -1,
                "final": math.nan, "max": math.nan, "nan": int(values.size)}

    i = int(np.nanargmin(values))
    return {
        "n": int(values.size),
        "min": float(values[i]),
        "min_at": int(steps[i]),
        "final": float(finals[i]),
        "max": float(values[finite].max()),
        "nan": int((~finite).sum()),
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

    table: dict[tuple[int, str], list[tuple[float, int]]] = defaultdict(list)
    derived: set[tuple[int, str]] = set()

    for budget in BUDGETS:
        for record in records:
            entry = best_at_budget(record, budget)
            if entry is None:
                continue
            key = (budget, record["label"])
            table[key].append(entry)
            if record["budget"] != budget:
                derived.add(key)

    print(
        f"\nProblem: {records[0]['problem']}   "
        f"seeds: {sorted({r['seed'] for r in records})}\n"
        "min = best over seeds; min_at/final = that seed's best step and last "
        "loss; max = worst seed\n"
    )

    rows = []
    for budget in BUDGETS:
        scored = [
            (summarize(entries)["min"], label, summarize(entries))
            for (b, label), entries in table.items()
            if b == budget
        ]
        if not scored:
            continue

        print(f"=== budget {budget} ===")
        print(f"{'algorithm':<46}{'n':>4}{'min':>11}{'min_at':>8}"
              f"{'final':>11}{'max':>11}{'nan':>5}")
        for _, label, stats in sorted(scored):
            tag = " (derived)" if (budget, label) in derived else ""
            print(
                f"{label + tag:<46}{stats['n']:>4}{stats['min']:>11.4f}"
                f"{stats['min_at']:>8}{stats['final']:>11.4f}"
                f"{stats['max']:>11.4f}{stats['nan']:>5}"
            )
            rows.append(
                {"budget": budget, "algorithm": label,
                 "derived": (budget, label) in derived, **stats}
            )

        # Headline: does the learned optimizer beat the best hand-designed one?
        velo = next((s for _, l, s in scored if l.startswith("velo")), None)
        baselines = [(m, l) for m, l, _ in scored if not l.startswith("velo")]
        if velo and baselines and math.isfinite(velo["min"]):
            best, best_label = min(baselines)
            verdict = "VeLO better" if velo["min"] < best else "baseline better"
            ratio = velo["min"] / best if best > 0 else math.nan
            print(
                f"  -> {verdict}: velo {velo['min']:.4f} vs "
                f"{best_label} {best:.4f}  (ratio {ratio:.2f}x)"
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

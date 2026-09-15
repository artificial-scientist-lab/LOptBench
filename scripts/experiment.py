"""Multi-seed three-way study: Adam vs NA-Adam vs VeLO on Voyager.

The three algorithms form a ladder of *scheduling*:

    Adam      no schedule                   budget is only a stopping rule
    NA-Adam   hand-designed noise anneal    budget-relative or fixed-length
    VeLO      learned schedule              budget is a hyperparameter

so the study asks something sharper than "who wins": whether VeLO's advantage
comes from *learning* the update rule, or merely from *having* an annealing
schedule at all.

Structure
---------
Stage "tune" selects hyperparameters at a single budget on **held-out seeds**
(100-102), so the reported results are not contaminated by choosing
hyperparameters on the same seeds they are scored on. Stage "study" runs the
selected configuration over all budgets and seeds 0-9.

VeLO has nothing to tune, which is its claim; the tuning cost of the baselines
is reported alongside that zero.

Budget handling
---------------
Adam's trajectory does not depend on the budget, so one run at the largest
budget contains every smaller one (take the running minimum over the first N
steps) -- `analyze.py` derives those for free. VeLO and budget-relative NA-Adam
have no such property and get one run per budget. Algorithms are tagged
`budget_independent` accordingly, and `verify_prefix` **tests** the claim rather
than trusting it.

Operation
---------
Every run writes one JSON named deterministically from its cell, and an
existing file is skipped -- so the job is resumable and crash-tolerant, and
`--shard i --n-shards N` lets a SLURM array split the grid.

Usage:
    python scripts/experiment.py --list                      # show the grid
    python scripts/experiment.py --stage tune
    python scripts/experiment.py --stage study --shard 0 --n-shards 20
    python scripts/experiment.py --verify-prefix
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------- grid config

PROBLEM = "voyager"

# These three are ITERATED, so they must be tuples. For a single value keep the
# trailing comma -- (1000,) is a tuple, (1000) is just 1000.
BUDGETS = (1000,)  # full study: (250, 500, 1000, 2000, 4000)
STUDY_SEEDS = (42,)  # full study: tuple(range(10))
TUNE_SEEDS = (41,)  # full study: (100, 101, 102)

# This one is a SINGLE value, so it must be a plain int -- enumerate_cells
# already wraps it as (TUNE_BUDGET,). A tuple here ends up as max_evals=(1000,)
# and fails deep inside dfbench with "unsupported operand type(s) for -".
TUNE_BUDGET = 1000

#: Longest budget; budget-independent algorithms are run only here.
MAX_BUDGET = max(BUDGETS)

NA_ANNEAL_FRACTION = 0.5
NA_ANNEAL_ITERS = 5000      #TODO: probably does not make sense if NA_ANNEAL_ITERS 
                            # much less than the budgets (max_evals), because then 
                            # we can have mostly only noise around(?)


@dataclass(frozen=True)
class Config:
    """One algorithm configuration: what to run, and how budget affects it."""

    algorithm: str  # "velo" | "adam" | "na_adam_fixed" | "na_adam_relative"
    hyperparameters: dict[str, Any] = field(default_factory=dict)
    #: True if the trajectory is independent of the budget, so a single run at
    #: MAX_BUDGET yields every smaller budget by truncation.
    budget_independent: bool = False

    @property
    def label(self) -> str:
        if not self.hyperparameters:
            return self.algorithm
        parts = [f"{k}={v:g}" if isinstance(v, float) else f"{k}={v}"
                 for k, v in sorted(self.hyperparameters.items())]
        return f"{self.algorithm}[{','.join(parts)}]"


def tune_configs() -> list[Config]:
    """Hyperparameter grid for the baselines. VeLO has nothing to tune."""
    configs = [
        Config("adam", {"learning_rate": lr}, budget_independent=True)
        for lr in (0.003, 0.01, 0.03, 0.1, 0.3)
    ]
    for lr in (0.03, 0.1):
        for sigma in (0.1, 0.3, 1.0):
            configs.append(
                Config(
                    "na_adam_fixed",
                    {"learning_rate": lr, "noise_std_start": sigma},
                    budget_independent=True,
                )
            )
            configs.append(
                Config(
                    "na_adam_relative",
                    {"learning_rate": lr, "noise_std_start": sigma},
                )
            )
    return configs


def study_configs(selected: dict[str, dict[str, Any]]) -> list[Config]:
    """Best config per algorithm, plus VeLO.

    Args:
        selected: algorithm -> hyperparameters, as written by `--select`.
    """
    configs = [Config("velo")]
    for algorithm, hparams in sorted(selected.items()):
        configs.append(
            Config(
                algorithm,
                hparams,
                budget_independent=algorithm in _BUDGET_INDEPENDENT,
            )
        )
    return configs


_BUDGET_INDEPENDENT = {"adam", "na_adam_fixed"}

# ------------------------------------------------------------------ execution


def build_algorithm(name: str):
    """Instantiate an algorithm. Imports are local so `--list` needs no JAX."""
    if name == "velo":
        from loptbench import VeLO

        return VeLO()
    if name == "adam":
        from dfbench.algorithms import OptaxAdam

        return OptaxAdam()
    if name in ("na_adam_fixed", "na_adam_relative"):
        from dfbench.algorithms import NAAdamGD

        return NAAdamGD()
    raise ValueError(f"unknown algorithm: {name}")


def build_problem(name: str):
    from dfbench.problems import (
        ConstrainedVoyagerProblem,
        UIFOProblem,
        VoyagerProblem,
        VoyagerTuningProblem,
    )

    if name == "voyager":
        return VoyagerProblem()
    if name == "voyager_tuning":
        return VoyagerTuningProblem()
    if name == "constrained_voyager":
        return ConstrainedVoyagerProblem()
    if name == "uifo":
        # topology_seed MUST be pinned: bare UIFOProblem() builds a different
        # random topology (and a different n_params) on every construction.
        return UIFOProblem(size=3, topology_seed=42)
    raise ValueError(f"unknown problem: {name}")


def resolve_hyperparameters(config: Config, budget: int) -> dict[str, Any]:
    """Per-run keyword arguments for `optimize()`, including budget wiring."""
    hparams = dict(config.hyperparameters)

    if config.algorithm == "velo":
        # VeLO is conditioned on its horizon; tie it to the budget explicitly
        # rather than relying on the Objective default, so it is recorded.
        hparams["num_steps"] = budget
    elif config.algorithm == "na_adam_relative":
        hparams["noise_anneal_budget_fraction"] = NA_ANNEAL_FRACTION
    elif config.algorithm == "na_adam_fixed":
        hparams["noise_anneal_iters"] = NA_ANNEAL_ITERS

    return hparams


def run_name(config: Config, budget: int, seed: int) -> str:
    """Deterministic, filesystem-safe name for one grid cell."""
    safe = config.label.replace("[", "_").replace("]", "").replace(",", "_")
    safe = safe.replace("=", "").replace(".", "p")
    return f"{PROBLEM}__{safe}__b{budget}__s{seed}.json"


def execute(config: Config, budget: int, seed: int, problem,
            stage: str = "study") -> dict:
    """Run one cell and return its record."""
    from dfbench import Objective

    hparams = resolve_hyperparameters(config, budget)
    algorithm = build_algorithm(config.algorithm)

    obj = Objective(problem, verbose=0, max_evals=budget)
    start = time.time()
    algorithm.optimize(obj, random_seed=seed, **hparams)
    elapsed = time.time() - start

    history = np.asarray(obj.loss_history, dtype=float)
    return {
        "problem": PROBLEM,
        # Which stage produced this run. Both stages share one output
        # directory, and `--select` / `analyze.py` filter on this field.
        # Filtering on seed instead would silently break whenever TUNE_SEEDS
        # is edited without updating analyze.py's matching default.
        "stage": stage,
        "algorithm": config.algorithm,
        "label": config.label,
        "hyperparameters": {k: v for k, v in hparams.items()},
        "budget_independent": config.budget_independent,
        "budget": budget,
        "seed": seed,
        "best_loss": float(obj.best_loss),
        "final_loss": float(history[-1]),
        "best_at": int(np.argmin(history)),
        "eval_count": int(obj.eval_count),
        "seconds": elapsed,
        "loss_history": history.tolist(),
        "env": {
            "host": socket.gethostname(),
            "python": platform.python_version(),
        },
    }


def enumerate_cells(stage: str, selected: dict | None) -> list[tuple[Config, int, int]]:
    """The full grid for a stage, in a fixed order so sharding is stable."""
    if stage == "tune":
        configs, seeds, budgets = tune_configs(), TUNE_SEEDS, (TUNE_BUDGET,)
    elif stage == "study":
        if selected is None:
            raise SystemExit(
                "--stage study needs selected hyperparameters. Run\n"
                "  python scripts/experiment.py --stage tune\n"
                "  python scripts/experiment.py --select\n"
                "first, or pass --selection <file>."
            )
        configs, seeds, budgets = study_configs(selected), STUDY_SEEDS, BUDGETS
    else:
        raise ValueError(stage)

    cells = []
    for config in configs:
        # A budget-independent algorithm is run only at MAX_BUDGET; smaller
        # budgets are derived in analyze.py by truncating the loss history.
        cell_budgets = (max(budgets),) if config.budget_independent else budgets
        for budget in cell_budgets:
            for seed in seeds:
                cells.append((config, budget, seed))
    return cells


def check_velo_cache() -> None:
    """Fail early and clearly if VeLO's checkpoint is not already local.

    Compute nodes frequently have no outbound network. Without this, twenty
    array tasks each try to download the same 9 MB file and fail obscurely.
    """
    from loptbench.checkpoints import VELO_CHECKPOINT, default_cache_dir

    cache = Path(os.environ.get("LOPTBENCH_CACHE") or default_cache_dir())
    params = cache / VELO_CHECKPOINT / "params"
    if not params.exists():
        raise SystemExit(
            f"VeLO checkpoint missing at {params}.\n"
            "Compute nodes often have no internet. On a machine that does, run:\n"
            "  python -c 'from loptbench import ensure_velo_checkpoint as e; e()'\n"
            f"then copy {cache} to the cluster, or point $LOPTBENCH_CACHE at a "
            "shared filesystem path."
        )


# ------------------------------------------------------------------- commands


def cmd_list(stage: str, selected: dict | None) -> None:
    cells = enumerate_cells(stage, selected)
    total_evals = sum(budget for _, budget, _ in cells)
    print(f"stage={stage}  cells={len(cells)}  evaluations={total_evals:,}")
    by_label: dict[str, list[int]] = {}
    for config, budget, _ in cells:
        by_label.setdefault(config.label, []).append(budget)
    for label, budgets in sorted(by_label.items()):
        print(f"  {label:<48} runs={len(budgets):<4} evals={sum(budgets):,}")


def cmd_run(stage: str, selected: dict | None, out_dir: Path,
            shard: int, n_shards: int) -> None:
    cells = enumerate_cells(stage, selected)
    mine = [c for i, c in enumerate(cells) if i % n_shards == shard]
    out_dir.mkdir(parents=True, exist_ok=True)

    if any(config.algorithm == "velo" for config, _, _ in mine):
        check_velo_cache()

    print(f"shard {shard}/{n_shards}: {len(mine)} of {len(cells)} cells", flush=True)

    problem = build_problem(PROBLEM)
    done = skipped = 0
    for config, budget, seed in mine:
        path = out_dir / run_name(config, budget, seed)
        if path.exists():
            skipped += 1
            continue
        record = execute(config, budget, seed, problem, stage=stage)
        # Write atomically so an interrupted job never leaves a partial file
        # that a later resume would mistake for a finished run.
        tmp = path.with_suffix(".json.partial")
        tmp.write_text(json.dumps(record))
        tmp.replace(path)
        done += 1
        print(
            f"  {config.label:<44} b={budget:<5} s={seed:<4} "
            f"best={record['best_loss']:.4f} final={record['final_loss']:.4f} "
            f"{record['seconds']:.0f}s",
            flush=True,
        )
    print(f"shard {shard}: {done} run, {skipped} already present", flush=True)


def record_stage(record: dict) -> str:
    """Which stage produced a record.

    Records written before the `stage` field existed fall back to the old
    seed-membership rule.
    """
    stage = record.get("stage")
    if stage:
        return stage
    return "tune" if record["seed"] in TUNE_SEEDS else "study"


def cmd_select(out_dir: Path, selection_path: Path) -> None:
    """Pick the best hyperparameters per algorithm from the tuning stage."""
    records = [json.loads(p.read_text()) for p in sorted(out_dir.glob("*.json"))]
    tune = [r for r in records if record_stage(r) == "tune"]
    if not tune:
        raise SystemExit(
            f"no tuning records found in {out_dir}. Run --stage tune first "
            f"(legacy records are matched by seed against TUNE_SEEDS={TUNE_SEEDS})."
        )

    by_algo_label: dict[tuple[str, str], list[dict]] = {}
    for r in tune:
        by_algo_label.setdefault((r["algorithm"], r["label"]), []).append(r)

    best: dict[str, dict[str, Any]] = {}
    print("Tuning results (median best_loss over held-out seeds):\n")
    for algorithm in sorted({a for a, _ in by_algo_label}):
        rows = []
        for (algo, label), rs in by_algo_label.items():
            if algo != algorithm:
                continue
            median = float(np.median([r["best_loss"] for r in rs]))
            rows.append((median, label, rs[0]["hyperparameters"], len(rs)))
        rows.sort()
        for median, label, _, n in rows:
            print(f"  {label:<48} median={median:>10.4f}  (n={n})")
        # Keep only the hyperparameters the user chose, not budget wiring.
        chosen = dict(rows[0][2])
        for wiring in ("num_steps", "noise_anneal_iters",
                       "noise_anneal_budget_fraction"):
            chosen.pop(wiring, None)
        best[algorithm] = chosen
        print(f"  -> selected {rows[0][1]}\n")

    selection_path.write_text(json.dumps(best, indent=2))
    print(f"Wrote {selection_path}")


def cmd_verify_prefix(out_dir: Path) -> None:
    """Test -- not assume -- the prefix property the compute saving rests on.

    For a budget-independent algorithm, a run at budget N must equal the first
    N steps of the run at MAX_BUDGET with the same seed. If that fails, the
    algorithm needs an explicit run per budget.
    """
    problem = build_problem(PROBLEM)
    seed = STUDY_SEEDS[0]
    short_budget = 1000
    failures = 0

    for config in tune_configs():
        if not config.budget_independent:
            continue
        if config.hyperparameters.get("learning_rate") != 0.1:
            continue  # one representative config per algorithm is enough

        long_path = out_dir / run_name(config, MAX_BUDGET, seed)
        if not long_path.exists():
            print(f"  {config.label}: no {MAX_BUDGET}-eval run yet, skipping")
            continue
        long_history = np.asarray(json.loads(long_path.read_text())["loss_history"])

        short = execute(config, short_budget, seed, problem)
        short_history = np.asarray(short["loss_history"])
        head = long_history[: len(short_history)]
        ok = np.array_equal(short_history, head)
        failures += 0 if ok else 1
        detail = "" if ok else f"  max|diff|={np.abs(short_history - head).max():.3e}"
        print(f"  {config.label:<48} prefix {'MATCH' if ok else 'DIFFER'}{detail}")

    if failures:
        raise SystemExit(
            f"{failures} algorithm(s) failed the prefix check -- they cannot use "
            "the derived-budget shortcut and need a run per budget."
        )
    print("All budget-independent algorithms verified.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", default="study", choices=("tune", "study"))
    parser.add_argument(
        "--out-dir", type=Path, default=None,
        help="where run JSONs live. Default: results/<stage>, i.e. "
             "results/tune for --stage tune and results/study for the study. "
             "--select reads results/tune, --verify-prefix reads results/study.",
    )
    parser.add_argument("--selection", type=Path, default=Path("results/selection.json"))
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--n-shards", type=int, default=1)
    parser.add_argument("--list", action="store_true", help="print the grid and exit")
    parser.add_argument("--select", action="store_true",
                        help="choose best hyperparameters from tuning records")
    parser.add_argument("--verify-prefix", action="store_true",
                        help="test the budget-independence assumption")
    args = parser.parse_args()

    if not 0 <= args.shard < args.n_shards:
        raise SystemExit(f"--shard must be in [0, {args.n_shards})")

    def out_dir_for(stage: str) -> Path:
        """Explicit --out-dir wins; otherwise one directory per stage."""
        return args.out_dir or Path("results") / stage

    selected = (
        json.loads(args.selection.read_text()) if args.selection.exists() else None
    )

    if args.select:
        cmd_select(out_dir_for("tune"), args.selection)
    elif args.verify_prefix:
        cmd_verify_prefix(out_dir_for("study"))
    elif args.list:
        cmd_list(args.stage, selected)
    else:
        cmd_run(args.stage, selected, out_dir_for(args.stage),
                args.shard, args.n_shards)


if __name__ == "__main__":
    sys.exit(main())

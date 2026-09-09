"""VeLO vs Adam on the Voyager problem, side by side.

A mix of ``voyager_velo.py`` and dfbench's ``voyager_adam_gd.py``: same
problem, same budget, same seeds, one table out.

Both optimizers spend exactly one objective evaluation per step, so an equal
``max_evals`` is an equal budget. Both start from the same random point for a
given seed (each calls ``random_params_unbounded()`` after seeding with it), so
differences come from the update rule, not the starting position.

Reported per run:
    best   -- the lowest loss seen at ANY logged evaluation (a running minimum)
    final  -- the loss at the last evaluation
    @      -- which evaluation achieved `best`

best and final differ when an optimizer finds a good point and then wanders
away from it. Watch for that: VeLO does it on short budgets.
"""

from dfbench import Objective
from dfbench.algorithms import AdamGD
from dfbench.problems import VoyagerProblem

from loptbench import VeLO

MAX_EVALS = 2000
SEEDS = [42]
ADAM_LR = 0.1

problem = VoyagerProblem()
rows = []

for seed in SEEDS:
    for name, algorithm, hparams in [
        ("velo", VeLO(), {}),
        ("adam", AdamGD(), {"learning_rate": ADAM_LR}),
    ]:
        obj = Objective(problem, verbose=0, max_evals=MAX_EVALS)
        algorithm.optimize(obj, random_seed=seed, **hparams)

        rows.append(
            {
                "algorithm": name,
                "seed": seed,
                "best": float(obj.best_loss),
                "final": float(obj.loss_history[-1]),
                # Which evaluation produced the best loss (0-indexed).
                "at": int(min(range(len(obj.loss_history)),
                              key=lambda i: obj.loss_history[i])),
                "evals": int(obj.eval_count),
            }
        )

print(f"\nVoyagerProblem   max_evals={MAX_EVALS}   adam lr={ADAM_LR}\n")
print(f"{'algorithm':<10}{'seed':>6}{'best':>12}{'final':>12}{'best @':>9}")
for r in rows:
    print(
        f"{r['algorithm']:<10}{r['seed']:>6}{r['best']:>12.4f}"
        f"{r['final']:>12.4f}{r['at']:>9}"
    )

print()
for name in ("velo", "adam"):
    best = [r["best"] for r in rows if r["algorithm"] == name]
    print(f"{name:<10} best loss over {len(best)} seeds: "
          f"median {sorted(best)[len(best) // 2]:.4f}   min {min(best):.4f}")

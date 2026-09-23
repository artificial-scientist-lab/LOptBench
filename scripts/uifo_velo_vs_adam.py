"""VeLO vs Adam on the UIFO problem, side by side.
Reported per run:
    best   -- the lowest loss seen at ANY logged evaluation (a running minimum)
    final  -- the loss at the last evaluation
    @      -- which evaluation achieved `best`
"""

from dfbench import Objective
from dfbench.algorithms import AdamGD
from dfbench.problems import UIFOProblem

from loptbench import VeLO

MAX_EVALS = 2000
SEEDS = [42]
UIFO_SIZE = 3
ADAM_LR = 0.1
VELO_NUM_STEPS = 1000 # number of evaluation steps, have to sweep across different values to find the best one (TODO)

rows = []

for seed in SEEDS:
    problem = UIFOProblem(topology_seed=seed, size=UIFO_SIZE)

    for name, algorithm, hparams in [
        ("velo", VeLO(), {}), #("adam", AdamGD(), {"learning_rate": ADAM_LR}),
    ]:
        obj = Objective(problem,
			verbose=0,
			max_time=4*60*60,print_every=1000, display_mode="log"
		)
        algorithm.optimize(obj, random_seed=seed, num_steps = VELO_NUM_STEPS,
                            **hparams)

        rows.append(
            {
                "algorithm": name,
                "seed": seed,
                "best": float(obj.best_loss),
                "final": float(obj.loss_history[-1]),
                "at": int(min(range(len(obj.loss_history)),
                              key=lambda i: obj.loss_history[i])),
                "evals": int(obj.eval_count),
            }
        )

print(f"\nUIFOProblem size={UIFO_SIZE}   max_evals={MAX_EVALS}   adam lr={ADAM_LR}\n")
print(f"{'algorithm':<10}{'seed':>6}{'best':>12}{'final':>12}{'best @':>9}")
for r in rows:
    print(
        f"{r['algorithm']:<10}{r['seed']:>6}{r['best']:>12.4f}"
        f"{r['final']:>12.4f}{r['at']:>9}"
    )

print()
for name in ("velo",): # "adam" commented out
    best = [r["best"] for r in rows if r["algorithm"] == name]
    print(f"{name:<10} best loss over {len(best)} seeds: "
          f"median {sorted(best)[len(best) // 2]:.4f}   min {min(best):.4f}")

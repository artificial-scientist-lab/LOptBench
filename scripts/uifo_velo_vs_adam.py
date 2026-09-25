"""VeLO vs Adam on the UIFO problem, side by side.
Reported per run:
    best   -- the lowest loss seen at ANY logged evaluation (a running minimum)
    final  -- the loss at the last evaluation
    @      -- which evaluation achieved `best`
"""
import argparse

from dfbench import Objective
from dfbench.algorithms import AdamGD
from dfbench.problems import UIFOProblem

from loptbench import VeLO

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--seed", type=int, default=0)
parser.add_argument("-num_steps", "--num_steps", type=int, default=1000)
parser.add_argument("-c", "--checkpoint",help="Celo2 checkpoint file", 
                    default='/home/krenn/klz397/LOptBench/celo2-base/theta.state')
args = parser.parse_args()
seed = args.seed

MAX_EVALS = 2000 # IT IS USED
SEEDS = [0]
UIFO_SIZE = 3
ADAM_LR = 0.1
VELO_NUM_STEPS = args.num_steps # number of evaluation steps, have to sweep across different values to find the best one (TODO)

rows = []

for seed in SEEDS:
    problem = UIFOProblem(topology_seed=seed, size=UIFO_SIZE)

    for name, algorithm, hparams in [
        ("velo", VeLO(), {}), #("adam", AdamGD(), {"learning_rate": ADAM_LR}),
    ]:
        obj = Objective(problem,
			verbose=1,
			max_time=4*60*60,
            print_every=1000, 
            display_mode="log",
            save_params_history=True,
		)
        algorithm.optimize(obj, random_seed=seed, num_steps = VELO_NUM_STEPS,
                            **hparams)

print(f"\nUIFOProblem size={UIFO_SIZE}   velo_num_steps={VELO_NUM_STEPS} \n")
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

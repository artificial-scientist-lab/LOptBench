"""Run Celo2 on the UIFO problem.

Mirrors dfbench's own ``scripts/uifo_adam_gd.py``.
"""
import argparse

from dfbench import Objective
from dfbench.problems import UIFOProblem
from dfbench.algorithms import AdamGD


parser = argparse.ArgumentParser()
parser.add_argument("-s", "--seed", type=int, default=42)
args = parser.parse_args()
seed = args.seed

# TODO: the constants sweep around?
max_evals = 1000
weight_decay = 1e-1
learning_rate = 1e-2

problem = UIFOProblem(topology_seed=seed, size=3)
obj = Objective(
    problem,
    verbose=1,
    max_evals=max_evals,
    max_time=10,
    # print_every=1000,
    save_params_history=True,
    save_to_file_every=1000,
    display_mode="log",
)

optimizer = AdamGD()


# Run optimization - returns Objective instance
optimizer.optimize(
    obj,
    random_seed=42,
    learning_rate=0.1,
)

obj.save_run_data()

print("Best loss:")
print(f"    {obj.best_loss:.6f}")
print("Total evaluations:")
print(f"    {obj.eval_count}")
print("First parameters:")
print(f"    {obj.params_history_bounded[0]}")
print("Best parameters:")
print(f"    {obj.best_params_bounded}")
print("Seed:")
print(f"    {seed}")

"""Run Celo2 on the UIFO problem.

Mirrors dfbench's own ``scripts/uifo_adam_gd.py``.
"""
import argparse

from dfbench import Objective
from dfbench.problems import UIFOProblem

from loptbench import Celo2

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--seed", type=int, default=42)
parser.add_argument("-c", "--checkpoint",help="Celo2 checkpoint file", 
                    default='../celo2-base/theta.state')
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

# pretrained_params = load_checkpoint()
# celo2-base. On dfbench's flat parameter vector orthogonalization is a no-op
# anyway, so this only has to match the checkpoint's architecture.
Celo2(args.checkpoint, orthogonalize=False).optimize(
    obj,
    learning_rate=learning_rate,
    weight_decay=weight_decay,
    patience=None,
    random_seed=seed,
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

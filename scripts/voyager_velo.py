"""Run VeLO on the Voyager problem.

Mirrors dfbench's own ``scripts/voyager_adam_gd.py`` so the two are directly
comparable.
"""

from dfbench import Objective
from dfbench.problems import VoyagerProblem

from loptbench import VeLO

vp = VoyagerProblem()
obj = Objective(
    vp,
    verbose=1,
    max_evals=1000,
    print_every=100,
    save_params_history=True,
)

# VeLO takes its horizon from obj.max_evals. It has no learning rate.
VeLO().optimize(obj, random_seed=42)

print(f"\nBest loss: {obj.best_loss:.6f}")
print(f"Total evaluations: {obj.eval_count}")
print(f"First parameters: {obj.params_history_bounded[0]}")
print(f"Last parameters: {obj.params_history_bounded[-1]}")



"""LOptBench: learned optimizers on the Differometor benchmark.

Contributes learned-optimizer algorithms that plug into ``dfbench``'s
``OptimizationAlgorithm`` interface, so they can be run and scored alongside
its hand-designed baselines::

    from dfbench.problems import VoyagerProblem
    from dfbench import Objective
    from loptbench import VeLO

    obj = Objective(VoyagerProblem(), max_evals=2000)
    VeLO().optimize(obj, random_seed=42)
"""

from loptbench import _tf_stub  # noqa: F401  (must precede learned_optimization)

from loptbench.algorithms import VeLO
from loptbench.checkpoints import VELO_CHECKPOINT, ensure_velo_checkpoint
from loptbench.velo_loader import load_velo, optax_velo

__version__ = "0.1.0"

__all__ = [
    "VELO_CHECKPOINT",
    "VeLO",
    "ensure_velo_checkpoint",
    "load_velo",
    "optax_velo",
]

"""Learned-optimizer algorithms for the Differometor benchmark.

To add one, subclass ``LearnedOptimizerAlgorithm`` and implement
``_make_optimizer``; the driver loop is inherited. See ``velo.py``.
"""

from loptbench.algorithms.base import LearnedOptimizerAlgorithm
from loptbench.algorithms.celo2 import Celo2
from loptbench.algorithms.velo import VeLO

__all__ = ["Celo2", "LearnedOptimizerAlgorithm", "VeLO"]

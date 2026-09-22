"""VeLO, a pretrained learned optimizer, as a dfbench algorithm."""

from __future__ import annotations

from typing import Optional

from loptbench.algorithms.base import LearnedOptimizerAlgorithm
from loptbench.checkpoints import VELO_CHECKPOINT
from loptbench.velo_loader import MAX_TRAINING_STEPS, optax_velo, velo_base_lopt_fn


class VeLO(LearnedOptimizerAlgorithm):
    """VeLO (Metz et al., 2022), a meta-trained learned optimizer.

    VeLO replaces a hand-designed update rule with a small LSTM plus a
    per-parameter MLP, meta-trained on thousands of neural-network training
    problems. It has no learning rate: the step size is part of what was
    learned.

    Note that dfbench parameters are a single flat ``(n_params,)`` array,
    whereas VeLO was meta-trained on the multi-tensor parameter trees of neural
    networks. It handles rank-1 inputs without complaint (factored
    second-moment features simply switch off), but this is outside its
    meta-training distribution, and low-dimensional problems especially so.

    Reference:
        Metz et al., "VeLO: Training Versatile Learned Optimizers by Scaling Up"
        (2022). https://arxiv.org/abs/2211.09760

    Hyperparameters exposed through ``optimize()``:
        num_steps, weight_decay, grad_clip_norm, patience.
    """

    algorithm_str: str = "velo"
    max_training_steps: int = MAX_TRAINING_STEPS

    def __init__(
        self,
        checkpoint: str = VELO_CHECKPOINT,
        cache_dir: Optional[str] = None,
    ) -> None:
        """Initialize.

        Args:
            checkpoint: Name of the pretrained checkpoint under
                ``gs://gresearch/learned_optimization/pretrained_lopts/``.
                The default is the one upstream's ``prefab`` treats as "VeLO".
            cache_dir: Where to cache the downloaded checkpoint. Defaults to
                ``~/.cache/loptbench`` (override with ``$LOPTBENCH_CACHE``).
        """
        self.checkpoint = checkpoint
        self.cache_dir = cache_dir

    def _make_optimizer(self, num_steps: int, weight_decay: float = 0.0, **kwargs):
        """Build VeLO for this horizon.

        Args:
            num_steps: Horizon to condition on.
            weight_decay: Optional decay wrapped around VeLO; upstream reports
                small values (~1e-6) as stabilizing. 0.0 disables it.
        """
        del kwargs
        return optax_velo(
            num_steps,
            weight_decay=weight_decay,
            base_lopt_fn=velo_base_lopt_fn(self.checkpoint, self.cache_dir),
        )

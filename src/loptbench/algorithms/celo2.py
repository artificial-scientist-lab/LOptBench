"""Celo2, a pretrained learned optimizer, as a dfbench algorithm."""

from __future__ import annotations

import os

import jax
import optax

from loptbench.algorithms.base import LearnedOptimizerAlgorithm
from loptbench.celo2_optax import load_checkpoint, scale_by_celo2


class Celo2(LearnedOptimizerAlgorithm):
    """Celo2 (Moudgil et al., 2026), a meta-trained learned optimizer.

    A small per-parameter MLP over momentum, RMS and Adafactor features that
    outputs a normalized update *direction*. Unlike VeLO the step size is not
    learned: it comes from an ordinary learning rate chained after the MLP, as
    in the reference usage::

        scale_by_celo2 -> add_decayed_weights -> scale_by_learning_rate

    Celo2 is not conditioned on the training horizon, so ``num_steps`` only
    matters here if a learning-rate schedule uses it.

    dfbench parameters are a single flat ``(n_params,)`` array. For rank-1
    parameters Celo2's factored features and Newton-Schulz orthogonalization
    both switch off, so ``orthogonalize`` makes no difference on these problems
    (Celo2 and Celo2-base coincide).

    Reference:
        https://arxiv.org/abs/2602.19142

    Hyperparameters exposed through ``optimize()``:
        learning_rate, weight_decay, grad_clip_norm, patience.
    """

    algorithm_str: str = "celo2"

    def __init__(self, checkpoint: str | dict, **config) -> None:
        """Initialize.

        Args:
            checkpoint: Path to a flax-serialized Celo2 checkpoint, or an
                already-loaded theta dict.
            **config: Forwarded to ``Celo2Transformation`` (e.g.
                ``orthogonalize=False`` for Celo2-base). Must match the
                architecture the checkpoint was trained with.
        """
        if isinstance(checkpoint, (str, os.PathLike)):
            checkpoint = load_checkpoint(os.fspath(checkpoint), **config)
        self.theta = checkpoint
        self.config = config

    def _make_optimizer(
        self,
        num_steps: int,
        learning_rate: float | optax.Schedule = 1e-2,
        weight_decay: float = 0.0,
        **kwargs,
    ):
        """Build Celo2 followed by weight decay and the learning rate.

        Args:
            num_steps: Unused by Celo2 itself (it is not horizon-conditioned).
            learning_rate: Constant or optax schedule applied to the MLP's
                normalized direction.
            weight_decay: Decoupled weight decay. 0.0 disables it.
        """
        del num_steps, kwargs
        opt = optax.chain(
            scale_by_celo2(self.theta, **self.config),
            optax.add_decayed_weights(weight_decay),
            optax.scale_by_learning_rate(learning_rate),
        )
        # The driver loop calls update() eagerly; without jit every step
        # re-traces the haiku MLP.
        return optax.GradientTransformationExtraArgs(opt.init, jax.jit(opt.update))

    def _extra_args(self, loss) -> dict:
        """Celo2 consumes only the gradient, not the loss."""
        del loss
        return {}

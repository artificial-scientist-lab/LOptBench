"""Build pretrained VeLO from a local checkpoint, as an optax transformation.

This is a reimplementation of
``learned_optimization.research.general_lopt.prefab``, which we cannot import:
``prefab`` pulls in ``pretrained_optimizers`` -> ``opt_from_checkpoint`` ->
``outer_train``, i.e. the entire meta-*training* stack, which requires
``tensorflow_datasets`` and friends. None of that is needed to *run* an
already-trained optimizer.

Instead we do what ``opt_from_checkpoint`` does after it has parsed the gin
config, but with the config read off by hand. The shipped ``config.gin`` for
the default VeLO checkpoint says, in full::

    HyperV2.lstm_hidden_size = 512
    HyperV2.param_inits = 256
    HyperV2.use_bugged_loss_features = False

so the architecture is reproducible without gin. See :data:`VELO_HPARAMS`.
"""

from __future__ import annotations

import functools
import os
from typing import Any, Callable, Optional

from loptbench import _tf_stub  # noqa: F401  (must precede learned_optimization)

import chex
import jax
import numpy as onp
from learned_optimization import checkpoints as lo_checkpoints
from learned_optimization.optimizers import base as opt_base
from learned_optimization.optimizers import gradient_accumulator
from learned_optimization.optimizers import opt_to_optax
from learned_optimization.optimizers import optimizer_wrappers
from learned_optimization.outer_trainers import gradient_learner
from learned_optimization.research.general_lopt import hyper_v2

from loptbench.checkpoints import VELO_CHECKPOINT, ensure_checkpoint

#: Architecture of the default VeLO checkpoint, read from its ``config.gin``.
VELO_HPARAMS = {
    "lstm_hidden_size": 512,
    "param_inits": 256,
    "use_bugged_loss_features": False,
}

#: Horizon beyond which ``LearnedOptimizer`` starts accumulating gradients.
#: VeLO was meta-trained on inner problems of at most this length.
MAX_TRAINING_STEPS = 150_000


@functools.lru_cache(maxsize=None)
def load_velo(
    name: str = VELO_CHECKPOINT,
    cache_dir: Optional[str] = None,
) -> opt_base.Optimizer:
    """Load a pretrained VeLO, downloading the checkpoint if needed.

    Cached: deserializing ~9 MB of meta-parameters takes a few seconds, and
    ``optimize()`` is typically called once per random seed.
    """
    params_path = ensure_checkpoint(name, cache_dir=cache_dir) / "params"

    lopt = hyper_v2.HyperV2(**VELO_HPARAMS)
    theta = lopt.init(jax.random.PRNGKey(0))
    ckpt = gradient_learner.ParameterCheckpoint(theta, "", 0)
    ckpt = lo_checkpoints.load_state(str(params_path), ckpt)
    return lopt.opt_fn(ckpt.params)


def velo_base_lopt_fn(
    name: str = VELO_CHECKPOINT,
    cache_dir: str | os.PathLike[str] | None = None,
) -> Callable[[], opt_base.Optimizer]:
    """Return a zero-arg factory, matching prefab's ``base_lopt_fn`` protocol."""
    cache_key = str(cache_dir) if cache_dir is not None else None
    return functools.partial(load_velo, name, cache_key)


class LearnedOptimizer(opt_base.Optimizer):
    """Pretrained learned optimizer with horizon handling and weight decay.

    A port of ``prefab.LearnedOptimizer``. Wraps the base optimizer in gradient
    accumulation when the requested horizon exceeds what it was meta-trained
    for, and optionally in weight decay, which upstream reports as stabilizing
    (1e-6 is usually enough).

    Args:
        num_training_steps: The horizon the optimizer is conditioned on. VeLO
            adapts its behaviour to the *fraction* of training elapsed, so this
            must be the real budget, not a placeholder.
        weight_decay: Added only when > 0.
        max_training_steps: Horizon above which gradients are accumulated.
        base_lopt_fn: Zero-arg factory returning the pretrained optimizer.
    """

    def __init__(
        self,
        num_training_steps: int,
        weight_decay: float = 0.0,
        max_training_steps: int = MAX_TRAINING_STEPS,
        base_lopt_fn: Optional[Callable[[], opt_base.Optimizer]] = None,
    ):
        super().__init__()
        if num_training_steps <= 0:
            raise ValueError(
                f"num_training_steps must be positive, got {num_training_steps}"
            )
        base_lopt_fn = base_lopt_fn or velo_base_lopt_fn()

        self.opt = base_lopt_fn()
        if num_training_steps > max_training_steps:
            num_accumulate = int(onp.ceil(num_training_steps / max_training_steps))
            self.opt = gradient_accumulator.GradientAccumulator(
                self.opt, num_accumulate
            )
        if weight_decay > 0.0:
            self.opt = optimizer_wrappers.WeightDecayWrapper(self.opt, weight_decay)
        self.num_training_steps = num_training_steps

    def init(
        self,
        params: chex.ArrayTree,
        model_state: Any = None,
        num_steps: Optional[int] = None,
        key: Optional[chex.PRNGKey] = None,
        **kwargs,
    ):
        if num_steps is not None and num_steps != self.num_training_steps:
            raise ValueError(
                f"num_steps={num_steps} must match the horizon this optimizer "
                f"was constructed with ({self.num_training_steps})."
            )
        # Randomness is unused by these optimizers.
        if key is None:
            key = jax.random.PRNGKey(0)
        return self.opt.init(
            params,
            model_state=model_state,
            num_steps=self.num_training_steps,
            key=key,
            **kwargs,
        )

    def update(
        self,
        opt_state,
        grad,
        model_state: Any = None,
        key: Optional[chex.PRNGKey] = None,
        **kwargs,
    ):
        if key is None:
            key = jax.random.PRNGKey(0)
        return self.opt.update(
            opt_state, grad, model_state=model_state, key=key, **kwargs
        )

    def get_params(self, opt_state):
        return self.opt.get_params(opt_state)

    def get_state(self, opt_state):
        return self.opt.get_state(opt_state)

    def set_params(self, opt_state, params):
        return self.opt.set_params(opt_state, params)

    def name(self):  # pyrefly: ignore[bad-override]
        return "LearnedOptimizer"


def optax_velo(
    num_steps: int,
    weight_decay: float = 0.0,
    max_training_steps: int = MAX_TRAINING_STEPS,
    base_lopt_fn: Optional[Callable[[], opt_base.Optimizer]] = None,
):
    """VeLO as an ``optax.GradientTransformationExtraArgs``.

    The returned transformation's ``update`` requires the current loss::

        updates, state = tx.update(grads, state, params,
                                   extra_args={"loss": loss})

    ``extra_args`` is a keyword-only dict here, not ``**kwargs`` -- see
    ``learned_optimization.optimizers.opt_to_optax``.
    """
    opt = LearnedOptimizer(
        num_steps,
        weight_decay=weight_decay,
        max_training_steps=max_training_steps,
        base_lopt_fn=base_lopt_fn,
    )
    return opt_to_optax.opt_to_optax_opt(opt, num_steps=num_steps)

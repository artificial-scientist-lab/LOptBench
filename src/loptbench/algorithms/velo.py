"""VeLO, a pretrained learned optimizer, as a dfbench algorithm."""

from __future__ import annotations

from typing import Optional

import jax
import jax.numpy as jnp
import optax
from jaxtyping import Array, Float

from dfbench.core.algorithm import AlgorithmType, OptimizationAlgorithm
from dfbench.core.objective import Objective

from loptbench.checkpoints import VELO_CHECKPOINT
from loptbench.velo_loader import MAX_TRAINING_STEPS, optax_velo, velo_base_lopt_fn

# The NaN/Inf recovery below mirrors dfbench's OptaxAlgorithm. We deliberately
# do not subclass or import from it: `dfbench.algorithms.__init__` eagerly
# imports every algorithm in the package, which would drag torch and evox into
# this one, and we override `optimize` wholesale anyway. These three names are
# kept identical to `dfbench.algorithms.gradient_based.optax._common` so the
# recovery behaviour matches the other gradient-based baselines.
_MAX_NAN_STREAK: int = 20
_NAN_PERTURB_BASE: float = 1e-10


def _is_nonfinite(loss, grads) -> bool:
    """Return True if loss or any gradient entry is NaN or Inf."""
    return bool(not jnp.isfinite(loss) or not jnp.all(jnp.isfinite(grads)))


class VeLO(OptimizationAlgorithm):
    """VeLO (Metz et al., 2022), a meta-trained learned optimizer.

    VeLO replaces a hand-designed update rule with a small LSTM + per-parameter
    MLP that was meta-trained on thousands of neural-network training problems.
    It has no learning rate: the step size is part of what was learned.

    Two things distinguish it from a stock Optax optimizer, and both shape the
    loop below:

    1. **It consumes the loss**, not just the gradient. The loss enters through
       optax's ``extra_args`` channel, so the base class's ``update`` call is
       not usable and ``optimize`` is overridden.
    2. **It is conditioned on the training horizon.** VeLO adapts to the
       *fraction* of training elapsed, so the total step count must be known at
       ``optimize()`` time. It is taken from ``objective.max_evals`` unless
       given explicitly.

    Note that dfbench parameters are a single flat ``(n_params,)`` array,
    whereas VeLO was meta-trained on the multi-tensor parameter trees of neural
    networks. It handles rank-1 inputs without complaint (factored second-moment
    features simply switch off), but this is out of its meta-training
    distribution, and low-dimensional problems especially so.

    Reference:
        Metz et al., "VeLO: Training Versatile Learned Optimizers by Scaling Up"
        (2022). https://arxiv.org/abs/2211.09760

    Hyperparameters exposed through ``optimize()``:
        num_steps, weight_decay, grad_clip_norm, patience.
    """

    algorithm_str: str = "velo"
    algorithm_type: AlgorithmType = AlgorithmType.GRADIENT_BASED

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

    def optimize(
        self,
        objective: Objective,
        init_params: Float[Array, "..."] | None = None,
        random_seed: int | None = None,
        num_steps: int | None = None,
        weight_decay: float = 0.0,
        grad_clip_norm: float | None = None,
        patience: int | None = None,
        **kwargs,
    ) -> None:
        """Run VeLO for one horizon.

        Args:
            objective: Pre-configured Objective. Its ``max_evals`` supplies the
                horizon unless ``num_steps`` is given.
            init_params: Starting point. ``None`` -> random unbounded.
            random_seed: Seed for reproducibility.
            num_steps: Training horizon VeLO is conditioned on. Defaults to
                ``objective.max_evals``.
            weight_decay: Optional decay wrapped around VeLO; upstream reports
                small values (~1e-6) as stabilizing. 0.0 disables it.
            grad_clip_norm: Max global gradient norm, or None (default) to
                disable. Unlike the other Optax algorithms this defaults *off*:
                VeLO conditions on the raw gradient and does its own
                normalization internally.
            patience: Early-stop after this many evals without improvement.
            **kwargs: Ignored; accepted for interface compatibility.
        """
        obj = objective

        num_steps = num_steps if num_steps is not None else obj.max_evals
        if num_steps is None:
            raise ValueError(
                "VeLO is conditioned on the training horizon, so it needs one: "
                "pass num_steps=... to optimize(), or set max_evals=... on the "
                "Objective. A time-only budget is not enough."
            )
        if num_steps > MAX_TRAINING_STEPS:
            print(
                f"Horizon {num_steps} exceeds VeLO's meta-training length "
                f"({MAX_TRAINING_STEPS}); gradients will be accumulated."
            )

        random_seed, _ = self.prepare(obj, unbounded=True, random_seed=random_seed)

        if init_params is None:
            params = obj.random_params_unbounded() * (1 + 1e-8)
        else:
            params = init_params

        optimizer = optax_velo(
            num_steps,
            weight_decay=weight_decay,
            base_lopt_fn=velo_base_lopt_fn(self.checkpoint, self.cache_dir),
        )
        clip = (
            optax.clip_by_global_norm(grad_clip_norm)
            if grad_clip_norm is not None
            else None
        )

        # --- Warmup, before the clock starts ---------------------------------
        # VeLO's first update triggers a large JIT compile (its LSTM plus a
        # per-parameter MLP). Burn it here rather than against the time budget,
        # then discard the state so the real run starts from a clean LSTM.
        obj.warmup_value_and_grad()
        warm_state = optimizer.init(params)
        optimizer.update(
            jnp.zeros_like(params),
            warm_state,
            params,
            extra_args={"loss": jnp.ones(())},
        )
        del warm_state

        opt_state = optimizer.init(params)
        clip_state = clip.init(params) if clip is not None else None

        obj.start_logging()

        nan_streak = 0
        rng_key = jax.random.PRNGKey(random_seed if random_seed is not None else 0)

        while not obj.budget_exceeded:
            loss, grads = obj.value_and_grad(params)

            if patience is not None and obj.evals_since_improvement > patience:
                break

            # --- NaN / Inf guard, mirroring OptaxAlgorithm ---------------
            if _is_nonfinite(loss, grads):
                nan_streak += 1
                rng_key, sub_key = jax.random.split(rng_key)

                if nan_streak > _MAX_NAN_STREAK:
                    best = obj.best_params
                    if best is not None:
                        params = (
                            best
                            + jax.random.normal(sub_key, best.shape) * _NAN_PERTURB_BASE
                        )
                    else:
                        params = obj.random_params_unbounded()
                    opt_state = optimizer.init(params)
                    if clip is not None:
                        clip_state = clip.init(params)
                    nan_streak = 0
                else:
                    scale = _NAN_PERTURB_BASE * (2 ** min(nan_streak, 30))
                    params = params + jax.random.normal(sub_key, params.shape) * scale
                continue
            # --------------------------------------------------------------

            nan_streak = 0

            if clip is not None:
                # Applied by hand rather than via optax.chain: chaining drops
                # the extra_args channel VeLO needs.
                grads, clip_state = clip.update(grads, clip_state, params)

            updates, opt_state = optimizer.update(
                grads, opt_state, params, extra_args={"loss": loss}
            )
            params = optax.apply_updates(params, updates)

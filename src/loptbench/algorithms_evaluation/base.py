"""Shared driver loop for learned optimizers on dfbench problems.

Learned optimizers differ from hand-designed ones in two ways that the stock
``dfbench`` Optax driver cannot express, and both are handled here once:

1. **They consume the loss**, not just the gradient. It travels through optax's
   ``extra_args`` channel, which ``optax.chain`` does not forward -- hence the
   manual gradient clipping below.
2. **They are conditioned on the training horizon.** They adapt to the
   *fraction* of the budget elapsed, so the total step count must be known
   before the first step. Consequence: for a learned optimizer ``max_evals`` is
   a hyperparameter, not merely a stopping criterion, and runs at different
   budgets are not comparable.

Subclasses supply the optimizer and (rarely) the extra-args mapping:

    class MyLopt(LearnedOptimizerAlgorithm):
        algorithm_str = "my_lopt"

        def _make_optimizer(self, num_steps, **kwargs):
            return some_optax_transformation(num_steps)

Everything else -- horizon resolution, unbounded-space setup, JIT warmup off
the clock, NaN/Inf recovery, budget accounting -- is inherited.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import optax
from jaxtyping import Array, Float

from dfbench.core.algorithm import AlgorithmType, OptimizationAlgorithm
from dfbench.core.objective import Objective

# NaN/Inf recovery, mirroring dfbench's OptaxAlgorithm. We deliberately do not
# import from it: `dfbench.algorithms.__init__` eagerly imports every algorithm
# in the package, which would drag torch, evox and botorch into this module.
# These three names are kept identical to
# `dfbench.algorithms.gradient_based.optax._common` so that recovery behaviour
# matches the other gradient-based baselines.
_MAX_NAN_STREAK: int = 20
_NAN_PERTURB_BASE: float = 1e-10


def _is_nonfinite(loss, grads) -> bool:
    """Return True if loss or any gradient entry is NaN or Inf."""
    return bool(not jnp.isfinite(loss) or not jnp.all(jnp.isfinite(grads)))


class LearnedOptimizerAlgorithm(OptimizationAlgorithm):
    """Base class for pretrained learned optimizers as dfbench algorithms.

    Subclasses must set ``algorithm_str`` and implement ``_make_optimizer``.

    Class attributes:
        algorithm_str: Unique identifier, e.g. ``"velo"``.
        algorithm_type: Always ``GRADIENT_BASED`` -- these consume gradients.
        max_training_steps: Longest horizon the optimizer was meta-trained for,
            or None if unknown. Used only to warn on over-long budgets.
    """

    algorithm_type: AlgorithmType = AlgorithmType.GRADIENT_BASED
    max_training_steps: int | None = None

    # -- subclass hooks ------------------------------------------------------

    def _make_optimizer(self, num_steps: int, **kwargs):
        """Return the optimizer as an ``optax.GradientTransformationExtraArgs``.

        Args:
            num_steps: Horizon the optimizer should be conditioned on.
            **kwargs: Extra keywords forwarded from ``optimize()``.
        """
        raise NotImplementedError

    def _extra_args(self, loss) -> dict:
        """Per-step extras the optimizer needs beyond the gradient.

        Defaults to the loss, which is what VeLO and Celo consume. Override
        only for an optimizer with different requirements.
        """
        return {"loss": loss}

    # -- driver loop ---------------------------------------------------------

    def optimize(
        self,
        objective: Objective,
        init_params: Float[Array, "..."] | None = None,
        random_seed: int | None = None,
        num_steps: int | None = None,
        grad_clip_norm: float | None = None,
        patience: int | None = None,
        **kwargs,
    ) -> None:
        """Run the learned optimizer for one horizon.

        Args:
            objective: Pre-configured Objective. Its ``max_evals`` supplies the
                horizon unless ``num_steps`` is given.
            init_params: Starting point. ``None`` -> random unbounded.
            random_seed: Seed for reproducibility.
            num_steps: Horizon to condition on. Defaults to
                ``objective.max_evals``.
            grad_clip_norm: Max global gradient norm, or None (default) to
                disable. Unlike dfbench's Optax algorithms this defaults *off*:
                learned optimizers condition on the raw gradient and generally
                do their own normalization internally.
            patience: Early-stop after this many evals without improvement.
            **kwargs: Forwarded to ``_make_optimizer``.
        """
        obj = objective

        num_steps = num_steps if num_steps is not None else obj.max_evals
        if num_steps is None:
            raise ValueError(
                f"{type(self).__name__} is conditioned on the training horizon, "
                "so it needs one: pass num_steps=... to optimize(), or set "
                "max_evals=... on the Objective. A time-only budget is not "
                "enough."
            )
        if self.max_training_steps is not None and num_steps > self.max_training_steps:
            print(
                f"Horizon {num_steps} exceeds the meta-training length of "
                f"{self.algorithm_str} ({self.max_training_steps}); behaviour "
                "is extrapolated."
            )

        random_seed, _ = self.prepare(obj, unbounded=True, random_seed=random_seed)

        if init_params is None:
            params = obj.random_params_unbounded() * (1 + 1e-8)
        else:
            params = init_params

        optimizer = self._make_optimizer(num_steps, **kwargs)
        clip = (
            optax.clip_by_global_norm(grad_clip_norm)
            if grad_clip_norm is not None
            else None
        )

        # --- Warmup, before the clock starts ---------------------------------
        # The first update triggers a large JIT compile (an LSTM plus a
        # per-parameter MLP). Burn it here rather than against the time budget,
        # then discard the state so the real run starts from a clean network.
        obj.warmup_value_and_grad()
        warm_state = optimizer.init(params)
        optimizer.update(
            jnp.zeros_like(params),
            warm_state,
            params,
            extra_args=self._extra_args(jnp.ones(())),
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

            # --- NaN / Inf guard ------------------------------------------
            if _is_nonfinite(loss, grads):
                nan_streak += 1
                rng_key, sub_key = jax.random.split(rng_key)

                if nan_streak > _MAX_NAN_STREAK:
                    # Fallback: jump to best-known point or a fresh random start
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
                    # Escalating perturbation: starts at 1e-10 and doubles each
                    # consecutive miss (capped at 2**30).
                    scale = _NAN_PERTURB_BASE * (2 ** min(nan_streak, 30))
                    params = params + jax.random.normal(sub_key, params.shape) * scale
                continue
            # --------------------------------------------------------------

            nan_streak = 0

            if clip is not None:
                # Applied by hand rather than via optax.chain: chaining drops
                # the extra_args channel the optimizer needs.
                grads, clip_state = clip.update(grads, clip_state, params)

            updates, opt_state = optimizer.update(
                grads, opt_state, params, extra_args=self._extra_args(loss)
            )
            params = optax.apply_updates(params, updates)

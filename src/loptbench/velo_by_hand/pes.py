"""
supposed to follow https://github.com/google/learned_optimization/blob/main/learned_optimization/outer_trainers/truncated_pes.py

A vectorized, truncated, PES-Persistent evolution strategeis -based gradient estimator.

Pieces and their upstream counterparts
--------------------------------------
  TruncatedUnrollOut          truncated_step.TruncatedUnrollOut
  VectorizedTruncatedStep     truncated_step.VectorizedTruncatedStep
                              (+ LOpt truncated step: inner training + resets)
  vector_sample_perturbations common.vector_sample_perturbations
  truncated_unroll            common.truncated_unroll
  maybe_stacked_es_unroll     common.maybe_stacked_es_unroll
  compute_pes_grad            truncated_pes.compute_pes_grad
  PESWorkerState              truncated_pes.PESWorkerState
  TruncatedPES                truncated_pes.TruncatedPES
  meta_train                  gradient_learner's outer loop (AdamW on theta)
"""

from __future__ import annotations

import functools
from typing import Any, Callable, NamedTuple, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
from jax import lax
import optax

import velo_lopt as L

PRNGKey = jnp.ndarray
MetaParams = Any


# ---------------------------------------------------------------------------
# Small tree utilities (stand-ins for learned_optimization.tree_utils)
# ---------------------------------------------------------------------------
def tree_add(a, b):
    return jax.tree_util.tree_map(lambda x, y: x + y, a, b)


def tree_sub(a, b):
    return jax.tree_util.tree_map(lambda x, y: x - y, a, b)


def tree_zip_jnp(xs: Sequence[Any]):
    """List of identically-structured trees -> one tree with a new leading axis."""
    return jax.tree_util.tree_map(lambda *l: jnp.stack(l), *xs)


class PRNGSequence:
    """Minimal stand-in for hk.PRNGSequence: `next(rng)` yields fresh keys."""

    def __init__(self, key):
        self._key = key

    def __next__(self):
        self._key, sub = jax.random.split(self._key)
        return sub


# ---------------------------------------------------------------------------
# The inner problem
# ---------------------------------------------------------------------------
class Task(NamedTuple):
    """The inner problem the learned optimizer is meta-trained on."""
    init_params: Callable          # key -> list[jnp.ndarray]
    loss_fn: Callable              # (params, batch) -> scalar
    sample_batch: Callable         # key -> batch


class TruncatedUnrollOut(NamedTuple):
    """Per-step outputs of an unroll. Leaves are [num_tasks] (or [seq, num_tasks])."""
    loss: jnp.ndarray
    is_done: jnp.ndarray
    iteration: jnp.ndarray
    mask: jnp.ndarray


class InnerState(NamedTuple):
    """State of ONE inner training run."""
    params: Any
    opt_state: Any
    inner_step: jnp.ndarray
    horizon: jnp.ndarray


class VectorizedTruncatedStep:
    """Runs `num_tasks` inner training runs in parallel, one step at a time.

    Owns everything about the inner problem: initialization, one optimizer
    step, detecting the end of a run and resetting it. The estimator never
    sees params or optimizer state directly.
    """

    def __init__(self, task: Task, lopt: L.VeLO, num_tasks: int,
                 min_horizon: int, max_horizon: int,
                 use_mixing_layers: bool = False):
        if min_horizon < 1:
            raise ValueError("min_horizon must be >= 1")
        self.task = task
        self.lopt = lopt
        self.num_tasks = num_tasks
        self.min_horizon = min_horizon
        self.max_horizon = max_horizon
        self._lopt_update = (lopt.update_mixing_layers if use_mixing_layers
                             else lopt.update)
        self._grad_fn = jax.value_and_grad(task.loss_fn)

    # -- single task ---------------------------------------------------------
    def _single_init(self, key) -> InnerState:
        k1, k2 = jax.random.split(key)
        params = self.task.init_params(k1)
        return InnerState(
            params=params,
            opt_state=self.lopt.init_state(params),
            inner_step=jnp.asarray(0, dtype=jnp.int32),
            horizon=jax.random.randint(k2, (), self.min_horizon,
                                       self.max_horizon + 1, dtype=jnp.int32),
        )

    def _single_step(self, theta, state: InnerState, key, data):
        loss, grads = self._grad_fn(state.params, data)
        params, opt_state = self._lopt_update(
            theta, state.opt_state, state.params, grads, loss)
        inner_step = state.inner_step + 1
        is_done = inner_step >= state.horizon

        stepped = InnerState(params, opt_state, inner_step, state.horizon)
        fresh = self._single_init(key)
        new_state = jax.tree_util.tree_map(
            lambda f, s: jnp.where(is_done, f, s), fresh, stepped)

        out = TruncatedUnrollOut(
            loss=loss,
            is_done=is_done,
            iteration=inner_step,
            # The loss on the step that ends a run belongs to neither chain
            # cleanly, so it is masked out of the gradient estimate.
            mask=1.0 - is_done.astype(loss.dtype),
        )
        return new_state, out

    # -- vectorized ------------------------------------------------------------
    def init_step_state(self, theta, key) -> InnerState:
        del theta  # initial state does not depend on the meta-parameters
        return jax.vmap(self._single_init)(jax.random.split(key, self.num_tasks))

    def unroll_step(self, vec_theta, state, keys, data):
        """One inner step for every task. vec_theta has a leading task axis."""
        return jax.vmap(self._single_step)(vec_theta, state, keys, data)

    @functools.partial(jax.jit, static_argnums=(0, 2))
    def get_batch(self, key, steps: int):
        """Data with leading dims [steps, num_tasks]."""
        keys = jax.random.split(key, steps * self.num_tasks)
        keys = keys.reshape((steps, self.num_tasks) + keys.shape[1:])
        return jax.vmap(jax.vmap(self.task.sample_batch))(keys)


# ---------------------------------------------------------------------------
# common.* equivalents
# ---------------------------------------------------------------------------
def vector_sample_perturbations(theta, key, std, num_samples):
    """Returns (vec_pos, theta + vec_pos, theta - vec_pos), each [num_samples, ...]."""

    def _one(k):
        leaves, treedef = jax.tree_util.tree_flatten(theta)
        ks = jax.random.split(k, len(leaves))
        pos = jax.tree_util.tree_unflatten(treedef, [
            jax.random.normal(kk, l.shape, l.dtype) * std
            for kk, l in zip(ks, leaves)])
        return pos, tree_add(theta, pos), tree_sub(theta, pos)

    return jax.vmap(_one)(jax.random.split(key, num_samples))


@functools.partial(jax.jit, static_argnums=(0, 1, 2))
def truncated_unroll(truncated_step: VectorizedTruncatedStep, stack_factor: int,
                     steps: int, vec_theta, key, state, datas):
    """Unroll every task for `steps` steps under lax.scan.

    `steps` is passed in rather than measured off `datas`, because a
    deterministic task has `sample_batch -> None` and `datas` is then an empty
    pytree with no leaves to measure. Every dfbench problem is deterministic --
    there are no minibatches -- so that is the normal case here, not an edge
    case. Scanning over `(keys, None)` itself is fine; only the count was
    missing.
    """
    n = truncated_step.num_tasks
    keys = jax.random.split(key, steps * n)
    keys = keys.reshape((steps, n) + keys.shape[1:])
    if stack_factor == 2:  # stacked antithetic: both halves get identical keys
        keys = jnp.concatenate([keys, keys], axis=1)

    def body(state, xs):
        ks, data = xs
        return truncated_step.unroll_step(vec_theta, state, ks, data)

    return lax.scan(body, state, (keys, datas))


def maybe_stacked_es_unroll(truncated_step, stack_antithetic_samples: bool,
                            steps: int, vec_p_theta, vec_n_theta,
                            p_state, n_state, key, datas):
    """Unroll the positive and negative perturbations with shared keys and data.

    Stacked: concatenate both along the task axis and run ONE vmapped unroll.
    Unstacked: two separate unrolls. Same result; stacking trades memory for
    fewer kernel launches.
    """
    if not stack_antithetic_samples:
        p_state, p_ys = truncated_unroll(truncated_step, 1, steps, vec_p_theta,
                                         key, p_state, datas)
        n_state, n_ys = truncated_unroll(truncated_step, 1, steps, vec_n_theta,
                                         key, n_state, datas)
        return p_state, n_state, p_ys, n_ys

    cat0 = lambda a, b: jnp.concatenate([a, b], axis=0)
    cat1 = lambda d: jnp.concatenate([d, d], axis=1)
    theta = jax.tree_util.tree_map(cat0, vec_p_theta, vec_n_theta)
    state = jax.tree_util.tree_map(cat0, p_state, n_state)
    datas2 = jax.tree_util.tree_map(cat1, datas)

    state, ys = truncated_unroll(truncated_step, 2, steps, theta, key, state,
                                 datas2)

    n = truncated_step.num_tasks
    first = lambda x: x[:n]
    second = lambda x: x[n:]
    p_state = jax.tree_util.tree_map(first, state)
    n_state = jax.tree_util.tree_map(second, state)
    p_ys = jax.tree_util.tree_map(lambda y: y[:, :n], ys)
    n_ys = jax.tree_util.tree_map(lambda y: y[:, n:], ys)
    return p_state, n_state, p_ys, n_ys


# ---------------------------------------------------------------------------
# truncated_pes.compute_pes_grad -- ported near-verbatim
# ---------------------------------------------------------------------------
@functools.partial(jax.jit, static_argnames=("std", "sign_delta_loss_scalar"))
def compute_pes_grad(
    p_yses: Sequence[TruncatedUnrollOut],
    n_yses: Sequence[TruncatedUnrollOut],
    accumulator: MetaParams,
    vec_pos: MetaParams,
    std: float,
    sign_delta_loss_scalar: Optional[float] = None,
) -> Tuple[jnp.ndarray, MetaParams, MetaParams, TruncatedUnrollOut, jnp.ndarray]:
    """Compute the PES gradient estimate from the outputs of many unrolls.

    Returns (mean loss, es_grad, new_accumulator, p_ys, delta_losses).
    """

    def flat_first(x):
        return x.reshape([x.shape[0] * x.shape[1]] + list(x.shape[2:]))

    # [n_chunks, steps_per_jit, num_tasks] -> [seq, num_tasks]
    p_ys = jax.tree_util.tree_map(flat_first, tree_zip_jnp(p_yses))
    n_ys = jax.tree_util.tree_map(flat_first, tree_zip_jnp(n_yses))

    delta_losses = p_ys.loss - n_ys.loss

    if sign_delta_loss_scalar:
        # Robustness option: keep only the SIGN of the per-task delta loss.
        sign_per_task = jnp.sign(jnp.mean(delta_losses * p_ys.mask, axis=0))
        delta_losses = (jnp.ones_like(delta_losses) * sign_per_task *
                        sign_delta_loss_scalar)

    # True from the first step at which the inner problem finished onward.
    has_finished = lax.cumsum(jnp.asarray(p_ys.is_done, dtype=jnp.int32)) > 0

    denom = jnp.sum(p_ys.mask, axis=0)

    # losses from the chain that was running at the start of this truncation
    last_unroll_loss = jnp.sum(
        delta_losses * (1.0 - has_finished) * p_ys.mask, axis=0) / denom
    # losses from a chain that started during this truncation
    new_unroll_loss = jnp.sum(
        delta_losses * has_finished * p_ys.mask, axis=0) / denom

    factor = 1.0 / (2 * std**2)

    accumulator = tree_add(vec_pos, accumulator)

    num_tasks = last_unroll_loss.shape[0]

    def reshape_to(loss, p):
        return loss.reshape((num_tasks,) + (1,) * (len(p.shape) - 1)) * factor * p

    es_grad_from_accum = jax.tree_util.tree_map(
        functools.partial(reshape_to, last_unroll_loss), accumulator)
    es_grad_from_new_perturb = jax.tree_util.tree_map(
        functools.partial(reshape_to, new_unroll_loss), vec_pos)

    vec_es_grad = tree_add(es_grad_from_accum, es_grad_from_new_perturb)
    es_grad = jax.tree_util.tree_map(lambda x: jnp.mean(x, axis=0), vec_es_grad)

    def _switch_one_accum(a, b):
        shape = [num_tasks] + [1] * (len(a.shape) - 1)
        return jnp.where(jnp.reshape(has_finished[-1], shape), a, b)

    # a task that finished during this truncation restarts its chain at vec_pos
    new_accumulator = jax.tree_util.tree_map(_switch_one_accum, vec_pos,
                                             accumulator)

    pos_loss = jnp.sum(p_ys.loss * p_ys.mask, axis=0) / jnp.sum(p_ys.mask, axis=0)
    neg_loss = jnp.sum(n_ys.loss * n_ys.mask, axis=0) / jnp.sum(n_ys.mask, axis=0)

    return (jnp.mean((pos_loss + neg_loss) / 2.0), es_grad, new_accumulator,
            p_ys, delta_losses)


# ---------------------------------------------------------------------------
# The estimator
# ---------------------------------------------------------------------------
class WorkerWeights(NamedTuple):
    theta: MetaParams
    outer_state: Any = None        # kept for signature parity; unused here


class PESWorkerState(NamedTuple):
    pos_state: Any
    neg_state: Any
    accumulator: MetaParams


class UnrollInfo(NamedTuple):
    loss: jnp.ndarray
    iteration: jnp.ndarray
    is_done: jnp.ndarray


class GradientEstimatorOut(NamedTuple):
    mean_loss: jnp.ndarray
    grad: MetaParams
    unroll_state: PESWorkerState
    unroll_info: UnrollInfo


class TruncatedPES:
    """GradientEstimator for computing PES gradient estimates.

    Persistent Evolution Strategies keeps a running buffer of every
    perturbation applied to a still-running inner problem, which makes the
    truncated estimate unbiased (Vicol et al., 2021). Higher variance than
    plain truncated ES, lower bias.
    """

    def __init__(self,
                 truncated_step: VectorizedTruncatedStep,
                 trunc_length: int = 10,
                 std: float = 0.01,
                 steps_per_jit: int = 10,
                 stack_antithetic_samples: bool = False,
                 sign_delta_loss_scalar: Optional[float] = None):
        self.truncated_step = truncated_step
        self.std = std
        self.trunc_length = trunc_length
        self.steps_per_jit = steps_per_jit
        self.stack_antithetic_samples = stack_antithetic_samples
        self.sign_delta_loss_scalar = sign_delta_loss_scalar

        if self.trunc_length % self.steps_per_jit != 0:
            raise ValueError("Pass a trunc_length and steps_per_jit that are"
                             " multiples of each other.")

    def init_worker_state(self, worker_weights: WorkerWeights,
                          key: PRNGKey) -> PESWorkerState:
        theta = worker_weights.theta
        pos_unroll_state = self.truncated_step.init_step_state(theta, key)
        neg_unroll_state = pos_unroll_state      # identical start: common random numbers
        accumulator = jax.tree_util.tree_map(
            lambda x: jnp.zeros([self.truncated_step.num_tasks] + list(x.shape),
                                dtype=x.dtype),
            theta)
        return PESWorkerState(pos_state=pos_unroll_state,
                              neg_state=neg_unroll_state,
                              accumulator=accumulator)

    def get_datas(self, key):
        rng = PRNGSequence(key)
        return [self.truncated_step.get_batch(next(rng), self.steps_per_jit)
                for _ in range(self.trunc_length // self.steps_per_jit)]

    def compute_gradient_estimate(self, worker_weights: WorkerWeights,
                                  key: PRNGKey, state: PESWorkerState,
                                  datas_list: Optional[Sequence[Any]] = None
                                  ) -> GradientEstimatorOut:
        p_state = state.pos_state
        n_state = state.neg_state
        accumulator = state.accumulator
        rng = PRNGSequence(key)

        theta = worker_weights.theta

        vec_pos, vec_p_theta, vec_n_theta = vector_sample_perturbations(
            theta, next(rng), self.std, self.truncated_step.num_tasks)

        p_yses, n_yses = [], []
        for i in range(self.trunc_length // self.steps_per_jit):
            if datas_list is None:
                datas = self.truncated_step.get_batch(next(rng),
                                                      self.steps_per_jit)
            else:
                datas = datas_list[i]

            # force all to be non weak type -- for jit cache hits
            p_state = jax.tree_util.tree_map(
                lambda x: jnp.asarray(x, dtype=x.dtype), p_state)
            n_state = jax.tree_util.tree_map(
                lambda x: jnp.asarray(x, dtype=x.dtype), n_state)

            p_state, n_state, p_ys, n_ys = maybe_stacked_es_unroll(
                self.truncated_step, self.stack_antithetic_samples,
                self.steps_per_jit, vec_p_theta, vec_n_theta,
                p_state, n_state, next(rng), datas)

            p_yses.append(p_ys)
            n_yses.append(n_ys)

        loss, es_grad, new_accumulator, p_ys, _ = compute_pes_grad(
            p_yses, n_yses, accumulator, vec_pos, self.std,
            sign_delta_loss_scalar=self.sign_delta_loss_scalar)

        return GradientEstimatorOut(
            mean_loss=loss,
            grad=es_grad,
            unroll_state=PESWorkerState(p_state, n_state, new_accumulator),
            unroll_info=UnrollInfo(loss=p_ys.loss, iteration=p_ys.iteration,
                                   is_done=p_ys.is_done))


# ---------------------------------------------------------------------------
# Outer loop (the gradient_learner role)
# ---------------------------------------------------------------------------
def meta_train(task,
               key,
               hidden: int = 64,
               mlp_hidden: int = 4,
               n_pairs: int = 4,
               sigma: float = 0.01,
               trunc_len: int = 20,
               steps_per_jit: Optional[int] = None,
               inner_min: int = 40,
               inner_max: int = 200,
               num_steps: int = 200,
               outer_steps: int = 300,
               outer_lr: float = 3e-4,
               normalized_output: bool = True,
               use_mixing_layers: bool = False,
               stack_antithetic_samples: bool = False,
               sign_delta_loss_scalar: Optional[float] = None,
               log_every: int = 25,
               verbose: bool = True):
    """Meta-train theta with PES. Returns (theta, history_of_mean_loss).

    `task` may be a single Task or a list of Tasks. With several, one
    estimator is built per task and their gradients are averaged each outer
    step -- as upstream's gradient_learner does with multiple estimators. That
    also lets tasks have DIFFERENT architectures, since each estimator keeps
    its own particles.
    """
    tasks = [task] if isinstance(task, Task) else list(task)
    steps_per_jit = steps_per_jit or trunc_len

    lopt = L.VeLO(lstm_hidden_size=hidden, ff_hidden_size=mlp_hidden,
                  num_steps=num_steps, normalized_output=normalized_output)
    rng = PRNGSequence(key)
    theta = lopt.init_meta_params(next(rng))

    estimators = [
        TruncatedPES(
            VectorizedTruncatedStep(t, lopt, num_tasks=n_pairs,
                                    min_horizon=inner_min,
                                    max_horizon=inner_max,
                                    use_mixing_layers=use_mixing_layers),
            trunc_length=trunc_len, std=sigma, steps_per_jit=steps_per_jit,
            stack_antithetic_samples=stack_antithetic_samples,
            sign_delta_loss_scalar=sign_delta_loss_scalar)
        for t in tasks
    ]
    worker_states = [e.init_worker_state(WorkerWeights(theta), next(rng))
                     for e in estimators]

    outer_opt = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(optax.warmup_cosine_decay_schedule(
            1e-8, outer_lr, max(1, outer_steps // 20), outer_steps,
            outer_lr / 3)),
    )
    outer_state = outer_opt.init(theta)

    @jax.jit
    def apply_grad(theta, outer_state, grad):
        updates, outer_state = outer_opt.update(grad, outer_state, theta)
        return optax.apply_updates(theta, updates), outer_state

    history = []
    for i in range(outer_steps):
        outs = [e.compute_gradient_estimate(WorkerWeights(theta), next(rng), s)
                for e, s in zip(estimators, worker_states)]
        worker_states = [o.unroll_state for o in outs]
        grad = jax.tree_util.tree_map(lambda *g: sum(g) / len(g),
                                      *[o.grad for o in outs])
        theta, outer_state = apply_grad(theta, outer_state, grad)

        ml = float(sum(o.mean_loss for o in outs) / len(outs))
        history.append(ml)
        if verbose and (i % log_every == 0 or i == outer_steps - 1):
            print(f"  outer step {i:5d}   mean inner loss {ml:.4f}")

    return theta, history
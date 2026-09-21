"""Adapter: a dfbench problem as a :class:`pes.Task` for meta-training.

dfbench is used in two opposite ways in this repo, and it is worth being clear
about which is which:

* ``loptbench/algorithms/velo.py`` + ``base.py`` is the **evaluation** path.
  dfbench drives the loop -- the ``Objective`` counts evaluations, enforces the
  budget, logs history and computes ``best_loss`` -- and the optimizer is a
  passenger. That is how a finished optimizer gets *scored*.
* **Meta-training is the reverse.** We drive the loop, and dfbench has to be a
  pure ``params -> loss`` function that survives ``jit``, ``grad``, ``vmap``
  and ``lax.scan``. The ``Objective``'s bookkeeping is untraceable Python
  state and must be bypassed.

So this module does not "convert" dfbench into a task so much as reach past the
instrumentation for the one callable underneath it:
``obj.value_function(unbounded=True)``.

Usage::

    task = make_task(VoyagerProblem())
    phi, hist = pes.meta_train(task, key, num_steps=200, outer_steps=300)

Two things differ from ``demo.py``'s synthetic task, both structural:

1. **There are no batches.** dfbench problems are deterministic, so
   ``sample_batch`` returns ``None`` and the task distribution comes only from
   the initial parameter vector (plus, for UIFO, the topology). Meta-training
   on a single Voyager instance therefore produces a *specialist* for that one
   landscape -- a legitimate result, but not meta-generalization.
2. **The loss goes non-finite regularly**, and inside ``lax.scan`` there is no
   room for the recovery logic the evaluation path has. See
   :func:`_guarded_value_and_grad`.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable, Sequence

import jax
import jax.numpy as jnp
import numpy as np

import pes

#: Loss reported for a non-finite evaluation. Large enough that PES scores the
#: particle badly, small enough not to dominate the centred loss and turn one
#: bad step into the only thing the gradient estimate sees.
FALLBACK_LOSS = 1e6


def build_objective(problem, seed: int = 0, max_evals: int = 10**9):
    """Wrap a dfbench problem in an Objective configured for meta-training.

    The budget is set absurdly high deliberately: during meta-training *we*
    own the step count, and a real ``max_evals`` would make the Objective stop
    us mid-unroll. ``start_logging()`` is required before
    ``value_function()`` will hand anything back, but it records nothing unless
    ``obj(params)`` is called, which this module never does.
    """
    from dfbench import Objective

    obj = Objective(problem, max_evals=max_evals)
    obj.set_space_mode(unbounded=True)
    obj.set_seed(seed)
    obj.start_logging()
    return obj


def _guarded_value_and_grad(raw_loss: Callable) -> Callable:
    """NOT WIRED IN. Kept for when non-finite losses become a problem.

    dfbench losses go non-finite regularly -- that is what the recovery block
    in ``algorithms/base.py`` exists for. Meta-training has no equivalent: the
    unroll runs under ``lax.scan``, and one NaN loss flows through
    ``delta_losses`` into the whole ``es_grad``, leaving ``theta`` dead from
    that outer step onward with no error raised.

    A ``jnp.where`` inside the loss does **not** fix it -- the gradient still
    carries NaN from the branch that was not taken -- so a guard has to run
    *after* differentiation. That point is ``VectorizedTruncatedStep._grad_fn``
    in ``pes.py``; to enable this, pass it in there.

    A guarded step reports a large finite loss (the perturbation scores badly,
    so PES steers away) and a zero gradient (the inner optimizer does not move
    on garbage).

    Note the ``jnp.where`` on ``grad`` below assumes a single parameter tensor,
    which holds for dfbench but not in general -- use ``tree_map`` if this is
    ever reused elsewhere.
    """
    vg = jax.value_and_grad(raw_loss)

    def guarded(params, batch):
        del batch
        loss, grad = vg(params[0])
        bad = ~jnp.isfinite(loss)
        loss = jnp.where(bad, FALLBACK_LOSS, loss)
        grad = jnp.where(bad, 0.0, jnp.nan_to_num(grad, nan=0.0,
                                                  posinf=0.0, neginf=0.0))
        return loss, [grad]

    return guarded


def make_task(problem, seed: int = 0) -> pes.Task:
    """Build a :class:`pes.Task` from any dfbench problem.

    Args:
        problem: A dfbench problem instance, e.g. ``VoyagerProblem()`` or
            ``UIFOProblem(size=3, topology_seed=0)``.
        seed: Seeds the Objective's own sampler. Only relevant when
            ``init_params`` is called without an explicit key, which
            ``pes.meta_train`` never does.

    Returns:
        A Task whose parameters are a one-element list holding dfbench's flat
        ``(n_params,)`` vector -- ``velo_lopt.VeLO`` works on a *list* of
        tensors, and dfbench has exactly one.
    """
    obj = build_objective(problem, seed=seed)
    raw_loss = obj.value_function(unbounded=True)

    def init_params(key):
        return [obj.random_params_unbounded(rng_key=key)]

    def loss_fn(params, batch):
        del batch
        return raw_loss(params[0])

    def sample_batch(key):
        # dfbench problems are deterministic: no data, nothing to sample.
        del key
        return None

    return pes.Task(
        init_params=init_params,
        loss_fn=loss_fn,
        sample_batch=sample_batch,
    )


def make_uifo_grid(
    topology_seeds: Sequence[int] = (0, 1, 2, 3),
    size: int = 3,
) -> list[pes.Task]:
    """One Task per UIFO topology -- a genuine distribution over problems.

    Unlike Voyager, where the only randomness is the starting point, different
    ``topology_seed`` values give structurally different detectors, so a
    learned optimizer has something to generalize across.

    The catch, and why these cannot simply be concatenated into one Task:
    topologies have **different ``n_params``** (189 vs 195 observed). Different
    shapes cannot be ``vmap``ped together and each triggers its own XLA
    compile. Worse, a particle cannot migrate between topologies -- PES's
    persistent ``xi`` accumulation across truncations is the whole point of
    PES, and re-initializing a particle onto another problem throws it away.

    So a grid run needs one particle population *per* topology, with outer
    steps round-robining between them. ``pes.meta_train`` takes a single task
    and is not yet able to do that; see ``meta_train_grid`` below.
    """
    from dfbench.problems import UIFOProblem

    return [
        make_task(UIFOProblem(size=size, topology_seed=int(s)), seed=int(s))
        for s in topology_seeds
    ]


def describe(task: pes.Task, key=None) -> dict[str, Any]:
    """Sanity-check a task: parameter count, and the loss at a random start."""
    key = key if key is not None else jax.random.PRNGKey(0)
    params = task.init_params(key)
    loss, grad = jax.value_and_grad(task.loss_fn)(params, None)
    return {
        "n_params": int(params[0].size),
        "dtype": str(params[0].dtype),
        "loss_at_init": float(loss),
        "grad_norm": float(jnp.linalg.norm(grad[0])),
        "finite": bool(jnp.isfinite(loss)),
    }


# ---------------------------------------------------------------------------
# Saving a trained optimizer
# ---------------------------------------------------------------------------
def save_phi(path, phi, history, **config) -> Path:
    """Write meta-parameters, meta-loss history and the config beside them.

    ``phi`` alone is not enough to use a trained optimizer again: its array
    shapes depend on ``hidden``/``mlp_hidden``, and ``VeLO.update`` also needs
    ``num_steps`` and ``normalized_output`` to behave the way it did during
    meta-training. So the config travels with the weights.

    Writes ``<path>.npz`` (arrays) and ``<path>.json`` (config + history).
    """
    path = Path(path).with_suffix("")
    np.savez(path.with_suffix(".npz"), **{k: np.asarray(v) for k, v in phi.items()})
    path.with_suffix(".json").write_text(json.dumps(
        {"config": config, "history": [float(x) for x in history]}, indent=2))
    return path


def load_phi(path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Inverse of :func:`save_phi`. Returns ``(phi, meta)``."""
    path = Path(path).with_suffix("")
    with np.load(path.with_suffix(".npz")) as z:
        phi = {k: jnp.asarray(z[k]) for k in z.files}
    meta = json.loads(path.with_suffix(".json").read_text())
    return phi, meta


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------
def meta_train_voyager(
    key=None,
    num_steps: int = 200,
    outer_steps: int = 300,
    n_pairs: int = 4,
    trunc_len: int = 20,
    hidden: int = 64,
    mlp_hidden: int = 4,
    sigma: float = 0.01,
    outer_lr: float = 3e-4,
    normalized_output: bool = True,
    log_every: int = 25,
    use_mixing_layers: bool = True,
    out: str | None = "phi_voyager",
):
    """Meta-train the learned optimizer on VoyagerProblem.

    Cost, measured on an M-series Mac: one inner ``value_and_grad`` step is
    ~67 ms, and ``vmap`` over particles gives **no** speedup -- the simulator
    does not batch (8 particles: 19 evals/s vs 22 for one). So the cost is
    linear in particle count::

        outer_steps * 2*n_pairs * trunc_len  inner steps, at ~67 ms each

    The defaults are 300 x 8 x 20 = 48,000 steps, roughly **55 minutes**. Cut
    ``outer_steps`` and ``n_pairs`` first when trying things out.

    Args:
        out: Basename to save ``phi`` and the history under, or None to skip.

    Returns:
        ``(phi, history)`` -- the meta-parameters and the per-outer-step mean
        inner loss.
    """
    from dfbench.problems import VoyagerProblem

    key = key if key is not None else jax.random.PRNGKey(0)
    task = make_task(VoyagerProblem())
    print(f"Voyager task: {describe(task)}")

    inner = outer_steps * 2 * n_pairs * trunc_len
    print(f"{inner:,} inner steps at ~90 ms -> ~{inner * 0.090 / 60:.0f} min\n")

    t0 = time.time()
    phi, history = pes.meta_train(
        task, key,
        hidden=hidden, mlp_hidden=mlp_hidden,
        n_pairs=n_pairs, sigma=sigma, trunc_len=trunc_len,
        inner_min=max(trunc_len, num_steps // 4), inner_max=num_steps,
        num_steps=num_steps, outer_steps=outer_steps, outer_lr=outer_lr,
        normalized_output=normalized_output, log_every=log_every,
        use_mixing_layers=use_mixing_layers,
    )
    print(f"\n{time.time() - t0:.0f}s: meta-loss {history[0]:.4f} -> "
          f"{history[-1]:.4f} (min {min(history):.4f})")

    if out:
        p = save_phi(out, phi, history, problem="voyager", hidden=hidden,
                     mlp_hidden=mlp_hidden, num_steps=num_steps,
                     normalized_output=normalized_output,
                     use_mixing_layers=use_mixing_layers, outer_steps=outer_steps,
                     n_pairs=n_pairs, trunc_len=trunc_len, sigma=sigma)
        print(f"saved {p}.npz / {p}.json")
    return phi, history


def meta_train_uifo_grid(
    key=None,
    topology_seeds: Sequence[int] = (0, 1, 2, 3),
    size: int = 3,
    num_steps: int = 200,
    outer_steps: int = 400,
    n_pairs: int = 4,
    trunc_len: int = 20,
    hidden: int = 64,
    mlp_hidden: int = 4,
    sigma: float = 0.01,
    outer_lr: float = 3e-4,
    normalized_output: bool = True,
    log_every: int = 25,
    use_mixing_layers: bool = True,
    out: str | None = "phi_uifo_grid",
):
    """Meta-train across several UIFO topologies -- a real task distribution.

    Unlike Voyager, where the only randomness is the starting point, this gives
    the optimizer structurally different problems to generalize across. See
    :func:`make_uifo_grid` for why the topologies cannot share a particle
    population: ``pes.meta_train`` holds one population per task and
    round-robins outer steps between them.

    Because each topology is visited only every ``len(topology_seeds)`` outer
    steps, ``outer_steps`` should be scaled up accordingly.

    **Cost is unmeasured.** UIFO is much larger than Voyager (~190 parameters
    vs 48) and slow to construct. Run with two seeds and ``outer_steps=4``
    first to get a per-step number before committing.
    """
    key = key if key is not None else jax.random.PRNGKey(0)
    tasks = make_uifo_grid(topology_seeds, size=size)
    for s, t in zip(topology_seeds, tasks):
        print(f"UIFO topology_seed={s}: {describe(t)}")
    print()

    t0 = time.time()
    phi, history = pes.meta_train(
        tasks, key,
        hidden=hidden, mlp_hidden=mlp_hidden,
        n_pairs=n_pairs, sigma=sigma, trunc_len=trunc_len,
        inner_min=max(trunc_len, num_steps // 4), inner_max=num_steps,
        num_steps=num_steps, outer_steps=outer_steps, outer_lr=outer_lr,
        normalized_output=normalized_output, log_every=log_every,
        use_mixing_layers=use_mixing_layers,
    )
    print(f"\n{time.time() - t0:.0f}s: meta-loss {history[0]:.4f} -> "
          f"{history[-1]:.4f} (min {min(history):.4f})")

    if out:
        p = save_phi(out, phi, history, problem="uifo_grid",
                     topology_seeds=list(topology_seeds), size=size,
                     hidden=hidden, mlp_hidden=mlp_hidden, num_steps=num_steps,
                     normalized_output=normalized_output,
                     use_mixing_layers=mixinguse_mixing_layers_layers, outer_steps=outer_steps,
                     n_pairs=n_pairs, trunc_len=trunc_len, sigma=sigma)
        print(f"saved {p}.npz / {p}.json")
    return phi, history


if __name__ == "__main__":
    phi, hist = meta_train_voyager(outer_steps=300)
    print(f"\nmeta-loss: {hist[0]:.4f} -> {hist[-1]:.4f}")

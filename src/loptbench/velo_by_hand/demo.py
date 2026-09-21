"""
End-to-end demo:
  1. define an inner task (small MLP classifier)
  2. meta-train the learned optimizer on it with PES
  3. apply the frozen optimizer to a FRESH task and compare against tuned Adam

Replace `make_task(...)` with your own problem to train a learned optimizer for it.
"""

import jax
import jax.numpy as jnp
import numpy as np
import optax

import velo_lopt as L
import pes


# ---------------------------------------------------------------------------
# 1. Your problem goes here.
# ---------------------------------------------------------------------------
def make_task(n_in=16, n_hidden=32, n_out=4, batch=64, seed=0, n_data=2048):
    """Synthetic classification: a fixed teacher network labels random inputs."""
    tk = jax.random.PRNGKey(seed)
    k1, k2, k3 = jax.random.split(tk, 3)
    X = jax.random.normal(k1, (n_data, n_in))
    tW = jax.random.normal(k2, (n_in, n_out))
    Y = jnp.argmax(X @ tW + 0.1 * jax.random.normal(k3, (n_data, n_out)), axis=-1)

    def init_params(key):
        k1, k2 = jax.random.split(key)
        return [
            jax.random.normal(k1, (n_in, n_hidden)) / jnp.sqrt(n_in),
            jnp.zeros((n_hidden,)),
            jax.random.normal(k2, (n_hidden, n_out)) / jnp.sqrt(n_hidden),
            jnp.zeros((n_out,)),
        ]

    def loss_fn(params, b):
        W1, b1, W2, b2 = params
        x, y = b
        logits = jax.nn.relu(x @ W1 + b1) @ W2 + b2
        return optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()

    def sample_batch(key):
        idx = jax.random.randint(key, (batch,), 0, n_data)
        return (X[idx], Y[idx])

    return pes.Task(init_params, loss_fn, sample_batch)


# ---------------------------------------------------------------------------
# 2. Apply a frozen learned optimizer (this is how you use it after training)
# ---------------------------------------------------------------------------
def train_with_lopt(phi, task, key, n_steps, hidden=64, mlp_hidden=4,
                    normalized_output=True):
    opt = L.VeLO(lstm_hidden_size=hidden, ff_hidden_size=mlp_hidden,
                 num_steps=n_steps, normalized_output=normalized_output)
    params = task.init_params(key)
    state = opt.init_state(params)
    grad_fn = jax.jit(jax.value_and_grad(task.loss_fn))

    @jax.jit
    def step(params, state, key):
        key, sub = jax.random.split(key)
        loss, grads = grad_fn(params, task.sample_batch(sub))
        params, state = opt.update(phi, state, params, grads, loss)
        return params, state, key, loss

    losses = []
    for _ in range(n_steps):
        params, state, key, loss = step(params, state, key)
        losses.append(float(loss))
    return losses


def train_with_adam(task, key, n_steps, lr):
    params = task.init_params(key)
    opt = optax.adam(lr)
    st = opt.init(params)
    grad_fn = jax.jit(jax.value_and_grad(task.loss_fn))

    @jax.jit
    def step(params, st, key):
        key, sub = jax.random.split(key)
        loss, grads = grad_fn(params, task.sample_batch(sub))
        upd, st = opt.update(grads, st, params)
        return optax.apply_updates(params, upd), st, key, loss

    losses = []
    for _ in range(n_steps):
        params, st, key, loss = step(params, st, key)
        losses.append(float(loss))
    return losses


if __name__ == "__main__":
    N_STEPS = 200
    key = jax.random.PRNGKey(0)

    task = make_task(seed=0)

    print("Meta-training the learned optimizer with PES ...")
    key, k = jax.random.split(key)
    phi, hist = pes.meta_train(
        task, k, outer_steps=300, n_pairs=4, sigma=0.01,
        trunc_len=20, inner_min=40, inner_max=N_STEPS, num_steps=N_STEPS,
        outer_lr=3e-4, log_every=50)

    print("\nEvaluating on a FRESH initialization (held-out seed) ...")
    key, ke = jax.random.split(key)
    lo = train_with_lopt(phi, task, ke, N_STEPS)

    best, best_lr = None, None
    for lr in [3e-4, 1e-3, 3e-3, 1e-2, 3e-2]:
        a = train_with_adam(task, ke, N_STEPS, lr)
        if best is None or np.mean(a[-20:]) < np.mean(best[-20:]):
            best, best_lr = a, lr

    print(f"\n  learned optimizer  final loss: {np.mean(lo[-20:]):.4f}")
    print(f"  tuned Adam (lr={best_lr})  final loss: {np.mean(best[-20:]):.4f}")
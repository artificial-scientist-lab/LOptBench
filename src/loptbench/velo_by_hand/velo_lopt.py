"""
A VeLO-style hierarchical learned optimizer, as a single class.

Ports the supporting machinery from VeLO's `hyper_v2.py`:
  * factored_dims             -- pick the two largest axes for Adafactor
  * _clip_log_abs             -- log-magnitude compression for tensor features
  * _fractional_tanh_embed    -- HORIZON-RELATIVE time embedding
  * BufferLossAccumulators    -- bias-corrected, horizon-scaled loss features

    per-tensor LSTM  --(hypernetwork)-->  weights of a tiny per-parameter MLP
          ^                                          |
          | aggregate features                       | per-scalar features
          |                                          v
     one call per TENSOR                     one call per SCALAR  -> update

Meta-parameters (phi) = LSTM weights + hypernetwork head + input projection.
Everything else -- accumulators, LSTM hidden state, loss buffer -- is
non-learned runtime state, rebuilt from scratch on every training run.

Usage
-----
    opt   = VeLO(lstm_hidden_size=64, ff_hidden_size=4, num_steps=1000)
    phi   = opt.init_meta_params(key)      # meta-learned; freeze after training
    state = opt.init_state(params)         # fresh per run
    params, state = opt.update(phi, state, params, grads, loss)
"""

from __future__ import annotations

import functools
from typing import Any, NamedTuple, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as onp

EPS = 1e-8


# ---------------------------------------------------------------------------
# Helpers copied from hyper_v2.py
# ---------------------------------------------------------------------------
def factored_dims(shape: Sequence[int]) -> Optional[Tuple[int, int]]:
    """Whether to use a factored second moment estimator.

    Returns the two LARGEST axes to reduce over, or None for rank < 2 (in which
    case a full, unfactored second moment is used instead).
    NOTE: used for Adafactor
    """
    if len(shape) < 2:
        return None
    sorted_dims = onp.argsort(shape)
    return int(sorted_dims[-2]), int(sorted_dims[-1])


def _clip_log_abs(v, scale: float = 1.0):
    """Compress a wide-dynamic-range statistic into a bounded feature."""
    mag = jnp.log(1e-8 + jnp.abs(v * scale))
    return jnp.clip(mag, -5., 5.) * 0.5


def _fractional_tanh_embed(x):
    """Soft one-hot over FRACTION of training completed (x in [0, 1]).

    Unlike an absolute-step embedding this means the same thing on a 1K-step
    run and a 1M-step run, which is what lets the optimizer transfer horizons.
    """
    timescales = jnp.asarray([0.03, 0.1, 0.2, 0.4, 0.6, 0.8, 0.9, 1.0, 1.1],
                             dtype=jnp.float32)
    return jax.vmap(lambda t: jnp.tanh((x - t) * 10.))(timescales)


N_FRAC_FEATS = 9


def _safe_rsqrt(x):  # copied from hyper_v2.py !
    return jax.lax.rsqrt(jnp.maximum(x, 1e-9))


def _second_moment_normalizer(x, eps: float = 1e-5):
    """Scale a feature channel to second moment 1 across the tensor."""
    return x * jax.lax.rsqrt(eps + jnp.mean(jnp.square(x)))


# ---------------------------------------------------------------------------
class BufferLossAccumulators:
    """Rolling accumulator for loss values (copied from hyper_v2.py).

    Ten EMAs whose halflives are spread log-uniformly up to `num_steps`, so the
    timescales scale with the run length. Means are bias-corrected, incoming
    losses are clipped against a running estimate of scale, and the emitted
    feature says where the current loss sits between its running minimum and
    its slowest-moving average -- i.e. "am I still improving, or plateaued?"
    """

    n_decays = 10
    n_features = 9          # one is dropped at each end

    def init(self, num_steps):
        halflife = jnp.logspace(1, jnp.log10(num_steps), self.n_decays)
        decays = jnp.exp(-1. / halflife)
        return {
            "means": jnp.zeros((self.n_decays,), dtype=jnp.float32),
            "iteration": jnp.asarray(0, dtype=jnp.int32),
            "running_min": 999999999999. * jnp.ones((self.n_decays,),
                                                    dtype=jnp.float32),
            "decays": decays,
        }

    def update(self, state, loss):
        jdecays = state["decays"]
        cor_mean = state["means"] / (1 - jdecays ** (state["iteration"] + 1))
        approx_max = jnp.max(cor_mean)
        approx_max = jnp.where(state["iteration"] == 0, loss, approx_max)
        # clip so a diverging loss cannot wreck the buffer
        loss = jnp.minimum(jnp.abs(approx_max) * 2, loss)

        means = state["means"] * jdecays + loss * (1. - jdecays)
        cor_mean = means / (1 - jdecays ** (state["iteration"] + 1))
        running_min = jnp.minimum(state["running_min"], cor_mean)

        return {"means": means,
                "iteration": state["iteration"] + 1,
                "running_min": running_min,
                "decays": state["decays"]}

    def features(self, state):
        jdecays = state["decays"]
        cor_mean = state["means"] / (
            1 - jdecays ** jnp.maximum(state["iteration"], 1))
        approx_max = cor_mean[1:]        # slower-moving neighbour
        cor_mean = cor_mean[0:-1]
        running_min = state["running_min"][0:-1]

        den = jnp.maximum(1e-8, (approx_max - running_min))
        feature = (cor_mean - running_min) / den - 1.0
        feature = jnp.clip(feature, -1., 1.)
        # the first couple of iterations are meaningless
        return jnp.where(state["iteration"] <= 2, feature * 0, feature)


# ---------------------------------------------------------------------------
# State containers
# ---------------------------------------------------------------------------
class TensorState(NamedTuple):
    """Non-learned accumulators + LSTM state, one per parameter tensor."""
    mom: jnp.ndarray            # (n_mom, *shape)
    rms: jnp.ndarray            # (n_rms, *shape)
    fac_r: jnp.ndarray          # factored: (n_fac, *shape\d0); else (n_fac,*shape)
    fac_c: Optional[jnp.ndarray]  # factored: (n_fac, *shape\d1); else None
    h: jnp.ndarray              # (H,)
    c: jnp.ndarray              # (H,)


class OptState(NamedTuple):
    tensors: Any
    loss_buffer: Any
    step: jnp.ndarray


# ---------------------------------------------------------------------------
class VeLO:
    """Hierarchical hypernetwork-based learned optimizer."""

    def __init__(
        self,
        lstm_hidden_size: int = 64,
        ff_hidden_size: int = 4,
        momentum_decays: Sequence[float] = (0.9, 0.99, 0.999),
        rms_decays: Sequence[float] = (0.999,),
        adafactor_decays: Sequence[float] = (0.9, 0.99, 0.999),
        step_mult: float = 1e-3,
        exp_mult: float = 1e-3,
        normalized_output: bool = False, # True - ceLO normalization, False - veLO
        mix_layers: bool = True,
        param_scale_mult: bool = False,
        param_scale_floor: float = 1e-3,
        num_steps: int = 1000,
    ):
        """
        Args:
          lstm_hidden_size: width of the per-tensor LSTM (VeLO's default is 128).
          ff_hidden_size:   width of the per-parameter MLP (VeLO uses 4).
          momentum_decays / rms_decays / adafactor_decays: accumulator EMA decays.
          step_mult / exp_mult: VeLO's lambda_1 / lambda_2, kept small so early
                            meta-training cannot take destructive steps.
          normalized_output: True  -> Celo2 style, RMS-normalize the direction.
                             False -> VeLO style, step_mult * d * exp(exp_mult*m).
          param_scale_mult: scale the step by sqrt(mean(p^2)), as VeLO does, so
                            updates are relative to weight magnitude. Off by
                            default: it shrinks steps, so a short meta-training
                            run has no time to compensate via log_lr.
          param_scale_floor: lower bound on that scale. Without it a zero-
                            initialized tensor (every bias) has scale 0, gets a
                            zero step, and stays zero for the whole run.
          num_steps:        training horizon. Sets the loss-buffer halflives and
                            the fraction-trained embedding, so it must match the
                            run you actually intend.
        """
        self.H = lstm_hidden_size
        self.ff_hidden_size = ff_hidden_size
        self.mix_layers = mix_layers
        self.mom_decays = tuple(momentum_decays)
        self.rms_decays = tuple(rms_decays)
        self.fac_decays = tuple(adafactor_decays)
        self.step_mult = step_mult
        self.exp_mult = exp_mult
        self.normalized_output = normalized_output
        self.param_scale_mult = param_scale_mult
        self.param_scale_floor = param_scale_floor
        self.num_steps = num_steps

        self.n_mom = len(self.mom_decays)
        self.n_rms = len(self.rms_decays)
        self.n_fac = len(self.fac_decays)
        self.loss_buffer_fns = BufferLossAccumulators()

        # per-scalar channels: p, g, rsqrt_rms, g*rsqrt_rms, mom, mom*rsqrt_rms,
        #                      fac-normalized g, fac-normalized mom
        self.n_param_feats = (2 + self.n_rms + self.n_rms * (1 + self.n_mom)
                              + 2 * self.n_fac)
        self.feat_dim = self.n_param_feats + N_FRAC_FEATS

    # --------------------------------------------------------- meta-params
    def tensor_feat_dim(self) -> int:
        # fraction_left + loss_features + var_m + mean_rms + var_rms + rank
        return (N_FRAC_FEATS + BufferLossAccumulators.n_features
                + self.n_mom + 2 * self.n_rms + 5)

    def mlp_weight_count(self) -> int:
        f, k = self.feat_dim, self.ff_hidden_size
        return f * k + k + k * 2 + 2

    def init_meta_params(self, key):
        """Initialize phi. This is the ONLY thing meta-training changes."""
        k1, k2, k3, k4, k5 = jax.random.split(key, 5)
        in_dim, n_w = self.tensor_feat_dim(), self.mlp_weight_count()

        def glorot(k, shape, gain=1.0):
            return jax.random.normal(k, shape) * (gain / jnp.sqrt(shape[0]))

        return {
            "proj_w": glorot(k1, (in_dim, self.H)),
            "proj_b": jnp.zeros((self.H,)),
            "lstm_ih": glorot(k2, (self.H, 4 * self.H)),
            "lstm_hh": glorot(k3, (self.H, 4 * self.H)),
            "lstm_b": jnp.concatenate([jnp.zeros((self.H,)),
                                       jnp.ones((self.H,)),
                                       jnp.zeros((2 * self.H,))]),
            "hyper_w": glorot(k4, (self.H, n_w), gain=0.1),
            "hyper_b": jnp.zeros((n_w,)),
            "lr_w": glorot(k5, (self.H, 1), gain=0.1),
            "lr_b": jnp.zeros((1,)),
        }

    # -------------------------------------------------------- accumulators
    def _update_accumulators(self, st: TensorState, g) -> TensorState:
        mom = jnp.stack([b * st.mom[i] + (1. - b) * g
                         for i, b in enumerate(self.mom_decays)])
        g2 = jnp.square(g)
        rms = jnp.stack([b * st.rms[i] + (1. - b) * g2
                         for i, b in enumerate(self.rms_decays)])

        fd = factored_dims(g.shape)
        if fd is None:
            fac_r = jnp.stack([b * st.fac_r[i] + (1. - b) * g2
                               for i, b in enumerate(self.fac_decays)])
            return st._replace(mom=mom, rms=rms, fac_r=fac_r)

        d1, d0 = fd
        row_g = jnp.mean(g2, axis=d0)     # drops axis d0
        col_g = jnp.mean(g2, axis=d1)     # drops axis d1
        fac_r = jnp.stack([b * st.fac_r[i] + (1. - b) * row_g
                           for i, b in enumerate(self.fac_decays)])
        fac_c = jnp.stack([b * st.fac_c[i] + (1. - b) * col_g
                           for i, b in enumerate(self.fac_decays)])
        return st._replace(mom=mom, rms=rms, fac_r=fac_r, fac_c=fac_c)

    # ------------------------------------------------------------ features
    def _per_param_features(self, p, g, st: TensorState, frac_feat):
        """
        (n_scalars, feat_dim) -- one row per scalar in the tensor.
        NOTE: implementation of _ff_mod(...) in original heyperv2
        """
        chans = [p, g]
        for j in range(self.n_rms):
            rsqrt_rms = _safe_rsqrt(st.rms[j] + EPS)
            chans.append(rsqrt_rms)
            chans.append(g * rsqrt_rms)
            chans += [st.mom[i] * rsqrt_rms for i in range(self.n_mom)]

        fd = factored_dims(g.shape)
        for i in range(self.n_fac):
            if fd is None:
                scale = _safe_rsqrt(st.fac_r[i])
            else:
                d1, d0 = fd
                row = jnp.expand_dims(st.fac_r[i], axis=d0)
                col = jnp.expand_dims(st.fac_c[i], axis=d1)
                # Adafactor reconstruction: v_hat = row (x) col / mean(row)
                v_hat = row * col / (jnp.mean(st.fac_r[i]) + 1e-9)
                scale = _safe_rsqrt(v_hat)
            chans.append(g * scale)
            chans.append(st.mom[i % self.n_mom] * scale)

        chans = [_second_moment_normalizer(ch) for ch in chans]
        F = jnp.stack([ch.reshape(-1) for ch in chans], axis=-1)
        return jnp.concatenate(
            [F, jnp.broadcast_to(frac_feat, (F.shape[0], N_FRAC_FEATS))],
            axis=-1)

    def lstm_features_for_tensor(self, p, st: TensorState, frac_feat,
                                 loss_features):
        """Fixed-size summary of one tensor -- this is what the LSTM sees.

        Normalizes by the tensor's own scale first, then compresses each
        statistic with _clip_log_abs so wildly different layers look comparable.
        """
        norm_mult = _safe_rsqrt(jnp.mean(jnp.square(p)))
        m = st.mom * norm_mult
        rms = st.rms * norm_mult

        axes = tuple(range(1, m.ndim))          # all but the decay axis
        mean_m = jnp.mean(m, axis=axes, keepdims=True)
        var_m = jnp.mean(jnp.square(m - mean_m), axis=axes)

        raxes = tuple(range(1, rms.ndim))
        mean_rms = jnp.mean(rms, axis=raxes)
        var_rms = jnp.mean(
            # NOTE: var_rms = jnp.mean(jnp.square(rms - mean_m), axis=leading_axis) # in original version - a bug!
            jnp.square(rms - mean_rms.reshape((-1,) + (1,) * (rms.ndim - 1))),
            axis=raxes)

        n_rank = int(onp.sum(onp.asarray(p.shape) > 1))
        rank = jax.nn.one_hot(n_rank, 5)

        return jnp.concatenate([
            frac_feat,                            # 9
            loss_features,                        # 9
            _clip_log_abs(var_m, scale=10.),      # n_mom
            _clip_log_abs(mean_rms, scale=10.),   # n_rms
            _clip_log_abs(var_rms, scale=10.),    # n_rms
            rank,                                 # 5
        ])

    # ---------------------------------------------------------------- LSTM
    def _lstm_step(self, phi, x, h, c):
        gates = x @ phi["lstm_ih"] + h @ phi["lstm_hh"] + phi["lstm_b"]
        i, f, g, o = jnp.split(gates, 4, axis=-1)
        i, f, o = jax.nn.sigmoid(i), jax.nn.sigmoid(f), jax.nn.sigmoid(o)
        c_new = f * c + i * jnp.tanh(g)
        return o * jnp.tanh(c_new), c_new

    # -------------------------------------------------------- hypernetwork
    def _unpack_mlp(self, w_flat):
        """Slice the hypernetwork's flat output into the MLP's weights.

        These are *activations* of the LSTM, not meta-learned parameters.
        """
        f, k = self.feat_dim, self.ff_hidden_size
        i = 0
        W1 = w_flat[i:i + f * k].reshape(f, k); i += f * k
        b1 = w_flat[i:i + k];                   i += k
        W2 = w_flat[i:i + k * 2].reshape(k, 2); i += k * 2
        b2 = w_flat[i:i + 2]
        return W1, b1, W2, b2

    # -------------------------------------------------------- MLP step
    def _apply_mlp_step(self, p, F, h, phi):
        """Hidden state -> hypernetwork -> per-parameter MLP -> new parameters.
 
        Shared by `update` and `update_mixing_layers`, so the two paths can
        only differ in how h was computed.
        """
        W1, b1, W2, b2 = self._unpack_mlp(h @ phi["hyper_w"] + phi["hyper_b"])
        log_lr = (h @ phi["lr_w"] + phi["lr_b"])[0]
 
        out = jax.nn.relu(F @ W1 + b1) @ W2 + b2
        d, mag = out[:, 0], out[:, 1]
 
        if self.normalized_output:      # Celo2 style
            d = d * _safe_rsqrt(jnp.mean(jnp.square(d)) + EPS)
            step_vec = d * jnp.exp(jnp.clip(log_lr, -8., 2.)) * self.step_mult
        else:                           # VeLO / Adafac MLP LOpt
            step_vec = self.step_mult * d * jnp.exp(
                jnp.clip(self.exp_mult * mag, -8., 8.))
            step_vec = step_vec * jnp.exp(jnp.clip(log_lr, -4., 4.))
 
        if self.param_scale_mult:
            param_scale = jnp.sqrt(jnp.mean(jnp.square(p)) + 1e-9)
            param_scale = jnp.maximum(param_scale, self.param_scale_floor)
            step_vec = step_vec * param_scale
 
        return p - step_vec.reshape(p.shape)

    # ------------------------------------------------------ one tensor step
    def _tensor_step(self, phi, p, g, st: TensorState, frac_feat, loss_features):
        st = self._update_accumulators(st, g)   # updates the accumulators at the tensor state

        F = self._per_param_features(p, g, st, frac_feat) # constructing per param features with time horizon of frac_feat

        agg = self.lstm_features_for_tensor(p, st, frac_feat, loss_features)
        x = jnp.tanh(agg @ phi["proj_w"] + phi["proj_b"])
        h, c = self._lstm_step(phi, x, st.h, st.c)
        
        return self._apply_mlp_step(p, F, h, phi), st._replace(h=h, c=c)

    # ------------------------------------------------------------ interface
    def init_state(self, params) -> OptState:
        """Fresh optimizer state for a new run. Everything starts at zero."""
        tensors = []
        for p in params:
            fd = factored_dims(p.shape)
            if fd is None:
                fac_r = jnp.zeros((self.n_fac,) + p.shape)
                fac_c = None
            else:
                d1, d0 = fd
                rshape = p.shape[:d0] + p.shape[d0 + 1:]
                cshape = p.shape[:d1] + p.shape[d1 + 1:]
                fac_r = jnp.zeros((self.n_fac,) + rshape)
                fac_c = jnp.zeros((self.n_fac,) + cshape)
            tensors.append(TensorState(
                mom=jnp.zeros((self.n_mom,) + p.shape),
                rms=jnp.zeros((self.n_rms,) + p.shape),
                fac_r=fac_r, fac_c=fac_c,
                h=jnp.zeros((self.H,)), c=jnp.zeros((self.H,))))
        return OptState(tensors=tensors,
                        loss_buffer=self.loss_buffer_fns.init(self.num_steps),
                        step=jnp.array(0, dtype=jnp.int32))

    @functools.partial(jax.jit, static_argnums=(0,))
    def update(self, phi, state: OptState, params, grads, loss):
        """One optimizer step. Returns (new_params, new_state)."""
        loss_buffer = self.loss_buffer_fns.update(state.loss_buffer, loss)
        loss_features = self.loss_buffer_fns.features(loss_buffer)

        fraction_trained = state.step.astype(jnp.float32) / float(self.num_steps)
        frac_feat = _fractional_tanh_embed(fraction_trained)

        new_params, new_tensors = [], []
        for p, g, st in zip(params, grads, state.tensors):
            new_p, new_st = self._tensor_step(
                phi, p, g, st, frac_feat, loss_features)
            new_params.append(new_p)
            new_tensors.append(new_st)

        return new_params, OptState(tensors=new_tensors,
                                    loss_buffer=loss_buffer,
                                    step=state.step + 1)

    @functools.partial(jax.jit, static_argnums=(0,))
    def update_mixing_layers(self, phi, state: OptState, params, grads, loss):
        """Like `update`, but with VeLO's cross-tensor mixing before the LSTM.
 
        The loop splits into three phases because the max-pool needs every
        tensor's aggregate features before ANY tensor's LSTM step runs.
        """
        loss_buffer = self.loss_buffer_fns.update(state.loss_buffer, loss)
        loss_features = self.loss_buffer_fns.features(loss_buffer)
        fraction_trained = state.step.astype(jnp.float32) / float(self.num_steps)
        frac_feat = _fractional_tanh_embed(fraction_trained)
 
        # ---- phase 1: per tensor, up to the aggregate features --------------
        sts, Fs, aggs = [], [], []
        for p, g, st in zip(params, grads, state.tensors):
            st = self._update_accumulators(st, g)          # the ONLY update
            sts.append(st)
            Fs.append(self._per_param_features(p, g, st, frac_feat))
            aggs.append(self.lstm_features_for_tensor(
                p, st, frac_feat, loss_features))
 
        # ---- phase 2: global -- mixing, then one batched LSTM call ----------
        A = jnp.stack(aggs)                                 # (n_tensors, in_dim)
        X = jnp.tanh(A @ phi["proj_w"] + phi["proj_b"])     # (n_tensors, H)
        if self.mix_layers:
            mixed = jax.nn.relu(A @ phi["mix_w"] + phi["mix_b"])
            X = X + jnp.max(mixed, axis=0, keepdims=True)   # pool over tensors
        h_all, c_all = self._lstm_step(
            phi, X,
            jnp.stack([s.h for s in sts]),
            jnp.stack([s.c for s in sts]))
 
        # ---- phase 3: per tensor, hypernetwork -> MLP -> step ---------------
        new_params, new_tensors = [], []
        for i, (p, F, st) in enumerate(zip(params, Fs, sts)):
            new_params.append(self._apply_mlp_step(p, F, h_all[i], phi))
            new_tensors.append(st._replace(h=h_all[i], c=c_all[i]))
 
        return new_params, OptState(tensors=new_tensors,
                                    loss_buffer=loss_buffer,
                                    step=state.step + 1)
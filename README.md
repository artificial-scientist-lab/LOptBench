# LOptBench
Benchmark for Learned Optimizers
# LOptBench

Learned optimizers (currently VeLO) on the
[Differometor Benchmark](https://github.com/artificial-scientist-lab/Differometor-Benchmark).

## Install

Python 3.11–3.13. The two `--no-deps` flags are required, not optional — see the
notes below.

```bash
conda create -n veloenv python=3.12
conda activate veloenv

# 1. The benchmark
pip install "dfbench[all] @ git+https://github.com/artificial-scientist-lab/Differometor-Benchmark"

# 2. learned_optimization's real dependencies
pip install dm-haiku flax gin-config chex

# 3. learned_optimization itself, without its stale dependency pins
pip install --no-deps "learned_optimization @ git+https://github.com/google/learned_optimization"

# 4. This package
pip install -e . --no-deps
```

- **`[all]` in step 1:** `dfbench/algorithms/__init__.py` imports its whole
  algorithm zoo eagerly, so importing even plain Adam needs torch, evox,
  nevergrad and botorch.
- **`--no-deps` in step 3:** `learned_optimization`'s `setup.py` pins 2021-era
  versions (`dm-haiku==0.0.5`, `dm-launchpad-nightly`) that no longer resolve.
  The library works fine with current releases; only the metadata is stale.
- **`--no-deps` in step 4:** stops a re-resolve from upgrading `jax` out from
  under `dfbench`.
- **TensorFlow is not needed**, despite `learned_optimization` importing it.
  `src/loptbench/_tf_stub.py` supplies a fake module covering the two
  import-only statements. If it ever causes trouble:
  `pip install tensorflow-cpu`.

VeLO's pretrained weights (9 MB) download automatically on first use into
`~/.cache/loptbench/`. Override with `$LOPTBENCH_CACHE`.

## Run

```bash
conda activate veloenv
cd /path/to/LOptBench

python scripts/voyager_velo.py          # VeLO on Voyager
python scripts/voyager_adam_gd.py       # dfbench's Adam, same shape
python scripts/voyager_velo_vs_adam.py  # both, same budget and seed, one table
```

Edit the constants at the top of each script (`max_evals`, `random_seed`,
learning rate) to change a run.
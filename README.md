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

# Core: all that demo.py, pes.py and velo_lopt.py import
pip install "jax>=0.9" "optax>=0.2.6" "numpy>=2.0"

# Only if you meta-train on Voyager/UIFO (dfbench_task.py)
pip install "dfbench[optax] @ git+https://github.com/artificial-scientist-lab/Differometor-Benchmark"

```

## Run

```bash
conda activate veloenv
cd /path/to/LOptBench

cd src/loptbench/velo_by_hand
# python demo.py           # synthetic MLP task: meta-train with PES, then compare to Adam
python dfbench_task.py   # meta-train on a dfbench problem
```

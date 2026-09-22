---
license: mit
library_name: optax
tags:
  - optimizer
  - learned-optimizer
  - meta-learning
  - jax
---

# Celo2-base

<p>
<a href="https://arxiv.org/abs/2602.19142"><img alt="Paper" src="https://img.shields.io/badge/arXiv-2602.19142-b31b1b.svg"></a>
<a href="https://github.com/amoudgl/celo2"><img alt="Code" src="https://img.shields.io/badge/GitHub-black?logo=github&logoColor=white&labelColor=grey"></a>
<a href="https://opensource.org/licenses/MIT"><img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-yellow.svg"></a>
</p>


Official pretrained weights for **Celo2-base** learned update rule: This variant uses the learned update rule for all parameters without any optimization harness. For better performance, see [celo2](https://huggingface.co/amoudgl/celo2) that uses Newton-Schulz orthogonalization and AdamW for biases/embeddings.


## Quickstart

Download checkpoint and install:
```bash
pip install git+https://github.com/amoudgl/celo2.git
hf download amoudgl/celo2-base --local-dir ./celo2-base
```

Use `load_checkpoint` method to fetch pretrained params from checkpoint path:
```python
from celo2_optax import load_checkpoint
pretrained_params = load_checkpoint('./celo2-base/theta.state')
``` 

Standard optax usage with `scale_by_celo2` method that takes pretrained params as input:
```python
import optax
from celo2_optax import scale_by_celo2

optimizer = optax.chain(
    scale_by_celo2(pretrained_params, orthogonalize=False),
    optax.add_decayed_weights(weight_decay),
    optax.scale_by_learning_rate(lr_schedule),
)
```

## Loading and inspecting MLP update rule weights

```python
from celo2_optax import load_checkpoint
import jax

pretrained_params = load_checkpoint('./celo2-base/theta.state')  # dictionary containing weights
print(jax.tree.map(lambda x: x.shape, pretrained_params))
```

The checkpoint contains a small MLP stored under the `ff_mod_stack` key with weight matrices (`w0__*`, `w1`, `w2`) and biases (`b0`, `b1`, `b2`). Each `w0__*` key contains weights corresponding to particular input feature such as momentum, gradient, parameter, etc.

## Meta-training config

| Key                     | Value                                                        |
| ----------------------- | ------------------------------------------------------------ |
| **Optimizer architecture** | MLP, 2 hidden layers, 8 units each                        |
| **Meta-training tasks**    | 4 image classification tasks (MNIST, FMNIST, CIFAR-10, SVHN) |
| **Task architecture**      | MLP (64-32-10)                             |
| **Meta-trainer**        | Persistent Evolution Strategies (PES)                        |
| **Outer iterations**    | 100K                                                         |
| **Truncation length**   | 50                                                           |
| **Min unroll length**   | 100                                                          |
| **Max unroll length**   | 2000                                                         |

For more details, see config JSON included in the repo [here](./config.json).

## Files

| File          | Description                      |
| ------------- | -------------------------------- |
| `theta.state` | Pretrained MLP optimizer weights |
| `config.json` | Meta-training configuration      |


## Citation

```bibtex
@misc{moudgil2026celo2,
      title={Celo2: Towards Learned Optimization Free Lunch},
      author={Abhinav Moudgil and Boris Knyazev and Eugene Belilovsky},
      year={2026},
      eprint={2602.19142},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2602.19142},
}
```

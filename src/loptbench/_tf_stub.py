"""Minimal stand-in for TensorFlow, registered only when TF is absent.

``learned_optimization.filesystem`` does a top-level ``import tensorflow as tf``
purely so that ``gs://`` paths can be routed through ``tf.io.gfile``. Every
local path falls through to the builtin ``open``/``os``/``shutil`` calls, so
nothing in the VeLO inference path actually touches TensorFlow.

Rather than pull in a ~500 MB dependency for an import statement, this module
registers a stub under ``sys.modules["tensorflow"]``. Importing it is a no-op
when a real TensorFlow is already installed.

One ordering subtlety: ``flax.io`` chooses its filesystem backend at import
time via ``importlib.util.find_spec("tensorflow")``, and would route all of its
file IO through our non-functional ``gfile`` if it saw the stub. So we import
``flax.io`` first, letting it settle on its native shim, and only then register
the stub. ``flax.io`` caches the decision in a module-level global, so the
later registration cannot disturb it.

Import this module *before* anything from ``learned_optimization``.
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
import sys
import types

_STUB_NAME = "tensorflow"

_GCS_HINT = (
    "This is loptbench's TensorFlow stub, not real TensorFlow. It was reached "
    "because something asked for a remote (gs://) path, which needs "
    "tf.io.gfile. Download the checkpoint locally instead -- see "
    "loptbench.checkpoints.ensure_velo_checkpoint() -- or install TensorFlow "
    "with `pip install tensorflow-cpu`."
)


class _StubGFile:
    """Every attribute resolves to a callable that raises with a hint."""

    def __getattr__(self, name: str):
        def _raise(*_args, **_kwargs):
            raise RuntimeError(f"tf.io.gfile.{name} was called. {_GCS_HINT}")

        return _raise


def _real_tensorflow_available() -> bool:
    if _STUB_NAME in sys.modules:
        return not getattr(sys.modules[_STUB_NAME], "__loptbench_stub__", False)
    try:
        return importlib.util.find_spec(_STUB_NAME) is not None
    except (ImportError, ValueError):
        return False


def _new_module(name: str) -> types.ModuleType:
    """A module with a real (if inert) spec.

    ``importlib.util.find_spec`` raises ``ValueError`` on a ``sys.modules``
    entry whose ``__spec__`` is None, and several libraries probe for
    TensorFlow that way.
    """
    mod = types.ModuleType(name)
    mod.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
    return mod


def _settle_flax_io_backend() -> None:
    """Let flax probe for TensorFlow before the stub exists."""
    try:
        import flax.io  # noqa: F401
    except ImportError:
        pass


def install() -> bool:
    """Register the stub if needed. Returns True if the stub is now in place."""
    if _STUB_NAME in sys.modules:
        # Either the real thing, or this stub from an earlier call.
        return getattr(sys.modules[_STUB_NAME], "__loptbench_stub__", False)

    if _real_tensorflow_available():
        return False

    _settle_flax_io_backend()

    tf = _new_module(_STUB_NAME)
    tf.__loptbench_stub__ = True
    tf.__version__ = "0.0.0+loptbench-stub"
    # Make it a package so `import tensorflow.compat.v2` resolves.
    tf.__path__ = []

    io = _new_module(f"{_STUB_NAME}.io")
    io.gfile = _StubGFile()
    tf.io = io

    # Anything that reaches the TF branch of a backend probe expects these.
    errors = _new_module(f"{_STUB_NAME}.errors")
    errors.NotFoundError = FileNotFoundError
    tf.errors = errors

    # `learned_optimization.summary` does `import tensorflow.compat.v2 as tf`
    # and calls `tf.enable_v2_behavior()` at module scope. Everything else it
    # needs from TF lives inside a Tensorboard writer we never construct.
    tf.enable_v2_behavior = lambda *_a, **_kw: None
    tf.constant = lambda value, *_a, **_kw: value
    tf.summary = _StubGFile()  # same raise-on-call behaviour, different name

    compat = _new_module(f"{_STUB_NAME}.compat")
    compat.__path__ = []
    compat.v1 = tf
    compat.v2 = tf
    tf.compat = compat

    sys.modules[_STUB_NAME] = tf
    sys.modules[f"{_STUB_NAME}.io"] = io
    sys.modules[f"{_STUB_NAME}.errors"] = errors
    sys.modules[f"{_STUB_NAME}.compat"] = compat
    sys.modules[f"{_STUB_NAME}.compat.v1"] = tf
    sys.modules[f"{_STUB_NAME}.compat.v2"] = tf
    return True


installed = install()

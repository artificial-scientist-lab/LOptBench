"""Fetch and cache pretrained learned-optimizer checkpoints.

``learned_optimization`` resolves its pretrained checkpoints against
``gs://gresearch/learned_optimization/pretrained_lopts/``, which needs a GCS
client (in practice, ``tf.io.gfile``). The same bucket is world-readable over
plain HTTPS, so we download the two files we need once and point the loader at
a local directory instead.

A checkpoint directory holds:

* ``params``     -- the serialized meta-parameters (~9 MB for VeLO)
* ``config.gin`` -- the operative gin config it was meta-trained under
"""

from __future__ import annotations

import os
import shutil
import urllib.error
import urllib.request
from pathlib import Path

_GCS_HTTP_ROOT = (
    "https://storage.googleapis.com/gresearch/learned_optimization/pretrained_lopts"
)

#: The checkpoint ``learned_optimization.research.general_lopt.prefab`` uses by
#: default, i.e. what "VeLO" means without further qualification.
VELO_CHECKPOINT = "aug12_continue_on_bigger_2xbs_200kstep_bigproblem_v2_5620"

_CHECKPOINT_FILES = ("params", "config.gin")

#: Expected size of VeLO's ``params``. Used only to detect a truncated
#: download; other checkpoints are not size-checked.
_VELO_PARAMS_BYTES = 9_282_564


def default_cache_dir() -> Path:
    """Root cache directory, overridable via ``LOPTBENCH_CACHE``."""
    env = os.environ.get("LOPTBENCH_CACHE")
    if env:
        return Path(env).expanduser()
    return Path.home() / ".cache" / "loptbench"


def _download(url: str, dest: Path) -> None:
    """Download to a temporary sibling, then move into place."""
    tmp = dest.with_suffix(dest.suffix + ".partial")
    try:
        with urllib.request.urlopen(url) as response, open(tmp, "wb") as out:
            shutil.copyfileobj(response, out)
    except urllib.error.HTTPError as exc:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"Could not download {url}: HTTP {exc.code}") from exc
    except OSError as exc:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"Could not download {url}: {exc}") from exc
    tmp.replace(dest)


def ensure_checkpoint(
    name: str = VELO_CHECKPOINT,
    cache_dir: str | os.PathLike[str] | None = None,
) -> Path:
    """Return a local directory holding ``name``'s checkpoint, downloading it once.

    Args:
        name: Checkpoint name under ``pretrained_lopts/``.
        cache_dir: Cache root. Defaults to :func:`default_cache_dir`.

    Returns:
        Path to the directory containing ``params`` and ``config.gin``.
    """
    root = Path(cache_dir).expanduser() if cache_dir else default_cache_dir()
    target = root / name
    target.mkdir(parents=True, exist_ok=True)

    for filename in _CHECKPOINT_FILES:
        path = target / filename
        if path.exists() and path.stat().st_size > 0:
            continue
        print(f"Downloading {name}/{filename} ...")
        _download(f"{_GCS_HTTP_ROOT}/{name}/{filename}", path)

    params = target / "params"
    if name == VELO_CHECKPOINT and params.stat().st_size != _VELO_PARAMS_BYTES:
        raise RuntimeError(
            f"{params} is {params.stat().st_size} bytes, expected "
            f"{_VELO_PARAMS_BYTES}. The download looks truncated -- delete the "
            "file and retry."
        )

    return target


def ensure_velo_checkpoint(
    cache_dir: str | os.PathLike[str] | None = None,
) -> Path:
    """Convenience wrapper: :func:`ensure_checkpoint` for the default VeLO."""
    return ensure_checkpoint(VELO_CHECKPOINT, cache_dir=cache_dir)

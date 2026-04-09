"""
Inference API: score 22-nt guide RNA sequences (DNA alphabet A/C/G/T).

Model artifacts (published upstream) are ``pri.h5`` and ``22nt.h5``.
Inspecting the saved graphs shows:

- ``pri.h5``: input shape ``(None, 56, 9, 1)`` (pri-miR-30-based encoding).
- ``22nt.h5``: **two** inputs — guide one-hot ``(None, 22, 4, 1)`` and a
  scalar ``(None, 1)`` matching the output of ``pri.h5``.

So the released runnable pipeline is the **two-stage** model (guide +
predicted pri surplus), not an isolated single-input gRNA-only checkpoint.
This module implements that published path without changing
:mod:`shRNAI.module_simple`.

Encoding for each guide matches the inner body of
:func:`shRNAI.module_simple.convert` (one-hot gRNA + pri tensor from the
fixed 97mer template).
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple, Union

import numpy as np

from shRNAI.module_simple import pair

# Lazy singletons for Keras models (tf_keras loads HDF5 checkpoints reliably).
_models: Optional[Tuple[Any, Any]] = None


def _package_root() -> Path:
    """Directory containing the installed ``shRNAI`` package (e.g. site-packages)."""
    return Path(__file__).resolve().parent.parent


def default_models_dir() -> Path:
    """Directory with ``pri.h5`` and ``22nt.h5``.

    Resolution order:

    1. Environment variable ``SHRNAI_MODELS_DIR`` (absolute path to a folder).
    2. ``<cwd>/models`` if both weight files exist there (typical for Colab after
       downloading weights into the working directory).
    3. ``<package_root>/models`` (editable / source checkout next to ``shRNAI/``).
    """
    env = os.environ.get("SHRNAI_MODELS_DIR")
    if env:
        return Path(env).expanduser().resolve()
    cwd_models = Path.cwd() / "models"
    if (cwd_models / "pri.h5").is_file() and (cwd_models / "22nt.h5").is_file():
        return cwd_models.resolve()
    return _package_root() / "models"


def _validate_guide(g: str, index: int) -> str:
    s = g.upper().replace("U", "T").strip()
    if len(s) != 22:
        raise ValueError(
            f"Guide at index {index} must have length 22, got {len(s)!r}"
        )
    for ch in s:
        if ch not in "ACGT":
            raise ValueError(
                f"Guide at index {index} must use DNA bases A/C/G/T "
                f"(or U), got {s!r}"
            )
    return s


def encode_guides(guides: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build ``onehotK`` and ``onehotK_pri`` arrays for a batch of 22-nt guides.

    Returns
    -------
    onehotK : ndarray, shape (N, 22, 4, 1)
    onehotK_pri : ndarray, shape (N, 56, 9, 1)
    """
    trans = str.maketrans("ACGT", "0123")
    trans2 = str.maketrans("ACGT", "TGCA")

    onehot_rows: List[np.ndarray] = []
    pri_rows: List[np.ndarray] = []

    for idx, raw in enumerate(guides):
        gRNA = _validate_guide(raw, idx)
        # pRNA = reverse complement of gRNA (same as convert()).
        pRNA = gRNA.translate(trans2)[::-1]
        # gRNA's 5' end should be unpaired (matches convert()).
        if gRNA[-1] in ["A", "T"]:
            pRNA = "C" + pRNA[1:]
        elif gRNA[-1] in ["C", "G"]:
            pRNA = "A" + pRNA[1:]

        var = list(map(int, list(gRNA.translate(trans))))
        var_onehot = np.eye(4)[var]
        onehot_rows.append(var_onehot)

        seq_pri = (
            "GGTATATTGCTGTTGACAGTGAGCG"
            + pRNA
            + "TAGTGAAGCCACAGATGTA"
            + gRNA
            + "TGCCTACTGCCTCGGAATTCAAGGG"
        )
        var_pri = list(map(int, list(seq_pri.translate(trans))))
        tempIn1 = np.delete(np.eye(5)[var_pri[:56]], -1, 1)
        tempIn2 = np.delete(np.eye(5)[var_pri[-56:][::-1]], -1, 1)
        tempIn3 = []
        for i in range(len(tempIn1)):
            seq1 = seq_pri[:56][i]
            seq2 = seq_pri[-56:][::-1][i]
            tempIn3.append(pair(seq1, seq2, i))
        seqIn = np.append(
            np.append(tempIn1, tempIn2, axis=1),
            np.asarray(tempIn3).reshape(len(tempIn3), 1),
            axis=1,
        )
        pri_rows.append(seqIn)

    onehotK = np.asarray(onehot_rows).reshape(-1, 22, 4, 1)
    onehotK_pri = np.asarray(pri_rows).reshape(-1, 56, 9, 1)
    return onehotK, onehotK_pri


def _load_models(models_dir: Path) -> Tuple[Any, Any]:
    global _models
    if _models is not None:
        return _models

    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import tf_keras as keras

    pri_path = models_dir / "pri.h5"
    nt_path = models_dir / "22nt.h5"
    if not pri_path.is_file():
        raise FileNotFoundError(f"Missing pri model: {pri_path}")
    if not nt_path.is_file():
        raise FileNotFoundError(f"Missing 22nt model: {nt_path}")

    model_pri = keras.models.load_model(pri_path, compile=False)
    model_22nt = keras.models.load_model(nt_path, compile=False)
    _models = (model_pri, model_22nt)
    return _models


def predict_potency(
    guides: Union[Sequence[str], List[str]],
    *,
    models_dir: Optional[Union[str, Path]] = None,
) -> np.ndarray:
    """
    Return potency scores (min–max normalized training scale) for each guide.

    Parameters
    ----------
    guides
        22-nt sequences using DNA alphabet (A/C/G/T); ``U`` maps to ``T``.
    models_dir
        Folder with ``pri.h5`` and ``22nt.h5``. Defaults to ``<repo>/models``.

    Returns
    -------
    ndarray, shape (N,)
        One float potency per input guide, same order as ``guides``.
    """
    md = Path(models_dir) if models_dir is not None else default_models_dir()
    onehotK, onehotK_pri = encode_guides(guides)
    model_pri, model_22nt = _load_models(md)

    pris = model_pri.predict(onehotK_pri, verbose=0).reshape(-1, 1)
    outs = model_22nt.predict([onehotK, pris], verbose=0).flatten()
    return outs


def _configure_tensorflow_cpu_only() -> None:
    """
    Force TensorFlow to use CPU only. Call before loading models (e.g. from
    ``main()``) so no GPU is visible when Keras loads weights.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    import tensorflow as tf

    try:
        tf.config.set_visible_devices([], "GPU")
    except Exception:
        pass
    try:
        tf.config.set_visible_devices([], "TPU")
    except Exception:
        pass


def main() -> None:
    """Benchmark CPU inference on 10 gRNA sequences (warmup + timed run)."""
    _configure_tensorflow_cpu_only()

    # Five 22-nt guides from upstream notebook output, duplicated to make 10.
    _five = [
        "ATAGTTTCAAACATCATCTTGT",
        "TTCATTGTCACTAACATCTGGT",
        "TTTTTCTGAGGTTTCCTCTGGT",
        "TACATCATCAATATTGTTCCTG",
        "TATATCTTCACCTTTAGCTGGC",
    ]
    guides = _five + _five

    predict_potency(guides)

    t0 = time.perf_counter()
    scores = predict_potency(guides)
    elapsed = time.perf_counter() - t0

    import tensorflow as tf

    devices = [d.name for d in tf.config.list_logical_devices()]
    print(f"TensorFlow logical devices: {devices}")
    print(
        f"CPU inference (10 guides, batch): {elapsed * 1000:.3f} ms total, "
        f"{elapsed / len(guides) * 1000:.3f} ms per guide"
    )
    for g, s in zip(guides, scores):
        print(f"{g}\t{float(s):.6f}")


if __name__ == "__main__":
    main()

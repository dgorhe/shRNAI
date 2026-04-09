# shRNAI

Convolutional neural network-based deep learning tool for predicting knockdown
efficiency of shRNAs.

## Inference (local MVP)

1. **Environment** (from repo root):

   ```bash
   uv venv
   uv sync
   ```

   For editors (mypy, Pylsp, autopep8, flake8): `uv sync --group dev`.

   Or: `uv pip install numpy tensorflow tf-keras` into your own venv.

2. **Model weights** — place `pri.h5` and `22nt.h5` under `models/` (default).
   Published files (same as [ParkSJ-91/shRNAI](https://github.com/ParkSJ-91/shRNAI)):

   - `https://raw.githubusercontent.com/ParkSJ-91/shRNAI/main/models/pri.h5`
   - `https://raw.githubusercontent.com/ParkSJ-91/shRNAI/main/models/22nt.h5`

3. **Score a Python list of 22-nt guide sequences** (DNA `A/C/G/T`; `U` → `T`):

   ```python
   from shRNAI.inference import predict_potency

   guides = ["ATAGTTTCAAACATCATCTTGT", "TTCATTGTCACTAACATCTGGT"]
   scores = predict_potency(guides)  # numpy 1d array, one float per guide
   ```

4. **CLI smoke test** (matches upstream `run_shRNAI.ipynb` example guides):

   ```bash
   uv run python -m shRNAI.inference
   ```

The published `22nt.h5` checkpoint expects **two** inputs (guide tensor + scalar
from `pri.h5`); see the module docstring in `shRNAI/inference.py`.

## Reference

- Paper: [Molecular Therapy: Nucleic Acids](https://doi.org/10.1016/j.omtn.2025.102738)
- Upstream notebook: `run_shRNAI.ipynb` in this repo (from GitHub)

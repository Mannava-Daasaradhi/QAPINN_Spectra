# Resolved Environment (T0.2)

Captured by running, inside the `uv`-managed venv:

```python
import sys, torch, pennylane, numpy, scipy
print('python:', sys.version)
print('torch:', torch.__version__)
print('torch.cuda:', torch.version.cuda)
print('cuda_available:', torch.cuda.is_available())
print('device_name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
print('pennylane:', pennylane.__version__)
print('numpy:', numpy.__version__)
print('scipy:', scipy.__version__)
```

## Output

```
python: 3.12.13 (main, Apr 14 2026, 14:31:26) [MSC v.1944 64 bit (AMD64)]
torch: 2.11.0+cu128
torch.cuda: 12.8
cuda_available: True
device_name: NVIDIA GeForce RTX 4090 Laptop GPU
pennylane: 0.45.1
numpy: 2.4.4
scipy: 1.18.0
```

## Full resolved dependency set (`uv add` output)

Runtime:
```
annotated-types==0.8.0
appdirs==1.4.4
autograd==1.8.0
autoray==0.8.4
cachetools==7.1.6
certifi==2022.12.7
charset-normalizer==2.1.1
colorama==0.4.6
contourpy==1.3.3
cycler==0.12.1
diastatic-malt==2.15.3
fonttools==4.63.0
gast==0.7.0
idna==3.4
iniconfig==2.3.0
kiwisolver==1.5.0
matplotlib==3.11.1
numpy==2.4.4
packaging==24.1
pandas==3.0.5
pennylane==0.45.1
pennylane-lightning==0.45.0
pillow==12.2.0
pluggy==1.6.0
pyarrow==25.0.0
pydantic==2.13.4
pydantic-core==2.46.4
pygments==2.20.0
pyparsing==3.3.2
pytest==9.1.1
python-dateutil==2.9.0.post0
pyyaml==6.0.3
requests==2.28.1
rustworkx==0.18.0
scipy==1.18.0
scipy-openblas32==0.3.34.0.0
six==1.17.0
termcolor==3.3.0
tomlkit==0.15.1
tqdm==4.66.5
typing-inspection==0.4.2
tzdata==2026.3
urllib3==1.26.13
torch==2.11.0+cu128 (from https://download.pytorch.org/whl/cu128)
```

Dev:
```
coverage==7.15.2
pytest-cov==7.1.0
ruff==0.16.0
```

## Notes

- Python **3.12.13**, installed via `uv python install 3.12` (project pins `requires-python = ">=3.12,<3.13"` per D10 — the miniconda base install (3.13.13) was left untouched).
- Torch wheel resolved from the `cu128` index (`https://download.pytorch.org/whl/cu128`), forward-compatible with the installed CUDA **13.0** driver (`nvidia-smi` reports `Driver Version: 581.80, CUDA Version: 13.0`) per D10's rationale.
- `torch.cuda.is_available()` is **True**; device is the machine's `NVIDIA GeForce RTX 4090 Laptop GPU` (16 GB VRAM), matching the hardware recorded in `00_MASTER_PLAN.md`.
- `pennylane==0.45.1` — this is the exact version cited in the master plan as the differentiable-oracle dependency (D2); it remains a hard dependency for `tests/test_qsim_vs_pennylane.py` (T2.5) even though the custom `qsim.py` fast path is used for training.
- Dependency management: `pyproject.toml` + `uv.lock`, both committed. No packages were added outside of this file (per the master plan's working rule 6).

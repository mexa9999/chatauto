# Optional AMD GPU inference benchmark

The simulation and its normal dependencies remain CPU-only. This benchmark
compares the existing NumPy policy with batched PyTorch inference on an AMD GPU
through DirectML; it is an experiment, not a production simulation backend.

The project requires Python 3.13+, but the current `torch-directml` wheels
support Python 3.12 (CPython ABI `cp312`) and earlier. Use a separate Python
3.12 environment; do not change the project's `.python-version` or install
DirectML into the main environment.

From the repository root in PowerShell:

```powershell
$gpuEnv = Join-Path $env:TEMP "citysim-gpu-benchmark"
uv venv --python 3.12 $gpuEnv
uv pip install --python (Join-Path $gpuEnv "Scripts\python.exe") numpy torch-directml
$env:PYTHONPATH = (Get-Location).Path
& (Join-Path $gpuEnv "Scripts\python.exe") tools\benchmark_gpu.py
```

The script creates independent CitySim worlds and observations, runs the
existing NumPy policy one world at a time, and compares it with a batched
DirectML implementation. GPU timings include feature transfer to the GPU and
result transfer back to the CPU, and exclude one-time device/model setup.
Outputs include a maximum absolute error check for scores, action logits, and
bid fractions.

To try a different batch range or repeat count:

```powershell
& (Join-Path $gpuEnv "Scripts\python.exe") tools\benchmark_gpu.py `
  --batch-sizes 16,32,64,128,256 --worlds 256 --repeats 20
```

DirectML/PyTorch can fall back to CPU for unsupported operators. The benchmark
does not suppress these warnings; inspect them when interpreting results. A
faster policy batch alone does not establish that the full simulator benefits:
world updates, policy batching, data transfer, and synchronization must also
be measured before integrating a GPU execution backend.

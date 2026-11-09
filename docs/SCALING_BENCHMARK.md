# CPU scaling benchmark

CitySim's scalable execution unit is an independent world/episode. Actions
inside one world remain sequential because each action can change the state
seen by subsequent agents. Training already runs independent episodes in
worker processes; population evaluation uses the same executor.

## Run a benchmark

From the repository root in PowerShell:

```powershell
.\.venv\Scripts\python.exe -m tools.benchmark_sim
```

By default, the benchmark checks 12×12 maps with 10 and 15 agents, 1/2/4
independent worlds, and up to 1/2/4 workers. It warms each executor before
timing, compares serial and process results for identical seeds, and reports
median elapsed time over three repeats. It uses six actions per agent and one
shared neural genome, so it measures execution throughput rather than strategy
quality. Process counts are capped at the number of simultaneous worlds.

To explore larger maps:

```powershell
.\.venv\Scripts\python.exe -m tools.benchmark_sim `
  --grids 20x20,32x32 --agents 20,30 --worlds 1,4 `
  --workers 2,4 --repeats 3 --actions 6
```

## Initial measurements

On the current Windows machine (16 logical CPUs, Python 3.13), the warmed
12×12, four-world benchmark showed:

| Agents | Serial | 4 workers | Speedup |
|---:|---:|---:|---:|
| 10 | 0.341 s | 0.093 s | 3.66× |
| 15 | 0.493 s | 0.142 s | 3.47× |

Two-worker speedups for four worlds were about 1.9×. One-world process runs
were approximately the same speed as serial for 12×12. A two-repeat exploratory
stress sample showed about 3.0–3.7× for four worlds at 20×20/20–30 agents and
about 2.9–3.2× at 32×32/20–30 agents; one-world process runs were slower on
the largest map. Every checked serial/process result matched.

These are machine-specific measurements, not guaranteed capacity limits. The
benchmark excludes process-pool startup and warms Numba before timing, matching
the long-lived pool used during training. Re-run with more repeats on the
intended machine before selecting worker defaults. More workers do not help a
single world; use process parallelism when there are enough independent,
compute-heavy episodes to amortize scheduling and serialization.

The optional [DirectML benchmark](GPU_BENCHMARK.md) measures policy inference
only, not full episodes; it does not establish a production GPU benefit.

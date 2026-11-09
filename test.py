from numba import njit
from time import perf_counter as pf

MOD = (1 << 31) - 1

@njit(cache=True)
def tick(x):
    for _ in range(9):
        x = (x * x) % MOD
    return x

@njit(cache=True)
def main(n=100_000):
    y = 1
    for x in range(1, n):
        y = (y * tick(x)) % MOD
    return y

def bench(fn, n=100_0000, runs=5):
    fn(n)                           # прогрев JIT
    times = []
    for _ in range(runs):
        st = pf()
        res = fn(n)
        times.append(pf() - st)
    return res, min(times), sum(times) / runs

if __name__ == "__main__":
    res, best, avg = bench(main)
    print(f"result = {res}, best = {best*1000:.2f} ms, avg = {avg*1000:.2f} ms")
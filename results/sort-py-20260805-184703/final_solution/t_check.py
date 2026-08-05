import random, time
import solution

# correctness
rnd = random.Random(42)
for trial in range(2000):
    n = rnd.randint(0, 300)
    lo = rnd.choice([-5, -100, 0, 10, -10**9, -2**63])
    hi = rnd.choice([5, 100, 1000, 10**9, 2**63])
    if hi < lo: lo, hi = hi, lo
    arr = [rnd.randint(lo, hi) for _ in range(n)]
    assert solution.sort_list(arr) == sorted(arr), (n, arr[:10])
for trial in range(100):
    n = rnd.randint(1000, 20000)
    arr = [rnd.randint(-2**63, 2**63) for _ in range(n)]
    assert solution.sort_list(arr) == sorted(arr)
    arr = [rnd.randint(0, 60) for _ in range(n)]
    assert solution.sort_list(arr) == sorted(arr)
# non-mutation
arr = [3,1,2]; c = list(arr); solution.sort_list(arr); assert arr == c
print("correct OK")

def bench(fn, arr, reps=11):
    fn(arr)
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); fn(arr); ts.append(time.perf_counter()-t0)
    ts.sort()
    return ts[len(ts)//2]

sizes = [500, 2000, 8000]
rnd = random.Random(7)
arrs = {sz: [rnd.randint(-(2**30), 2**30) for _ in range(sz)] for sz in sizes}
for sz in sizes:
    dt = bench(solution.sort_list, arrs[sz])
    print(sz, round(sz/dt/1000, 1))

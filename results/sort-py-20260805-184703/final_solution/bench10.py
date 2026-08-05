import random, time, bisect, heapq
from itertools import chain

def make_mergeheap(bisect_n, k, stride, cnt_mult=4, merge_max=6000):
    def v(arr):
        n = len(arr)
        if n < 2:
            return list(arr)
        if n <= bisect_n:
            out = []
            ins = bisect.insort
            for x in arr:
                ins(out, x)
            return out
        mn = min(arr); mx = max(arr)
        r = mx - mn + 1
        if r <= cnt_mult * n:
            cnt = [0] * r
            for x in arr:
                cnt[x - mn] += 1
            out = []
            for i, c in enumerate(cnt):
                if c:
                    out.extend([mn + i] * c)
            return out
        if n <= merge_max:
            if stride:
                chunks = [v(arr[i::k]) for i in range(k)]
            else:
                step = (n + k - 1) // k
                chunks = [v(arr[i*step:(i+1)*step]) for i in range(k)]
            return list(heapq.merge(*chunks))
        if mn < 0:
            a = [x - mn for x in arr]; mx -= mn
        else:
            a = list(arr)
        base = 2048 if n >= 8000 else 256
        mask = base - 1
        nb = base.bit_length() - 1
        shift = 0
        while mx >> shift:
            buckets = [[] for _ in range(base)]
            m = mask
            for x in a:
                buckets[(x >> shift) & m].append(x)
            a = list(chain.from_iterable(buckets))
            shift += nb
        if mn < 0:
            return [x + mn for x in a]
        return a
    return v

def make_radix(base_small=256, base_large=2048, large_n=8000):
    def v(arr):
        n = len(arr)
        if n < 2:
            return list(arr)
        if n <= 1024:
            out = []
            ins = bisect.insort
            for x in arr:
                ins(out, x)
            return out
        mn = min(arr); mx = max(arr)
        r = mx - mn + 1
        if r <= 4 * n:
            cnt = [0] * r
            for x in arr:
                cnt[x - mn] += 1
            out = []
            for i, c in enumerate(cnt):
                if c:
                    out.extend([mn + i] * c)
            return out
        if mn < 0:
            a = [x - mn for x in arr]; mx -= mn
        else:
            a = list(arr)
        base = base_large if n >= large_n else base_small
        mask = base - 1
        nb = base.bit_length() - 1
        shift = 0
        while mx >> shift:
            buckets = [[] for _ in range(base)]
            m = mask
            for x in a:
                buckets[(x >> shift) & m].append(x)
            a = list(chain.from_iterable(buckets))
            shift += nb
        if mn < 0:
            return [x + mn for x in a]
        return a
    return v

def check(fn):
    rnd = random.Random(42)
    for trial in range(300):
        n = rnd.randint(0, 100)
        lo = rnd.choice([-5, -100, 0, 10, -10**9])
        hi = rnd.choice([5, 100, 1000, 10**9, 10**9])
        if hi < lo:
            lo, hi = hi, lo
        arr = [rnd.randint(lo, hi) for _ in range(n)]
        if fn(arr) != sorted(arr):
            return False
    for trial in range(60):
        n = rnd.randint(100, 5000)
        arr = [rnd.randint(-(2**60), 2**60) for _ in range(n)]
        if fn(arr) != sorted(arr):
            return False
    for trial in range(60):
        n = rnd.randint(100, 5000)
        arr = [rnd.randint(0, 50) for _ in range(n)]
        if fn(arr) != sorted(arr):
            return False
    return True

def bench(fn, arr, reps=9):
    fn(arr)
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn(arr)
        ts.append(time.perf_counter() - t0)
    ts.sort()
    return ts[len(ts)//2]

def main():
    variants = {
        'cur': make_radix(),
        'm6s': make_mergeheap(1024, 6, True),
        'm7s': make_mergeheap(1024, 7, True),
        'm8s': make_mergeheap(1024, 8, True),
        'm9s': make_mergeheap(1024, 9, True),
        'm10s': make_mergeheap(1024, 10, True),
        'm12s': make_mergeheap(1024, 12, True),
        'm8c': make_mergeheap(1024, 8, False),
        'm16s': make_mergeheap(1024, 16, True),
        'm32s': make_mergeheap(1024, 32, True),
        'r1024': make_radix(1024, 2048, 8000),
        'r4096': make_radix(4096, 4096, 8000),
    }
    for name, fn in variants.items():
        assert check(fn), name

    for dname, gen in [('r30', lambda rnd, sz: [rnd.randint(-(2**30), 2**30) for _ in range(sz)]),
                       ('r63', lambda rnd, sz: [rnd.randint(-(2**63), 2**63) for _ in range(sz)])]:
        for sz in [2000, 8000]:
            acc = {n: [] for n in variants}
            for seed in range(7):
                rnd = random.Random(1000 + seed)
                arr = gen(rnd, sz)
                for name, fn in variants.items():
                    dt = bench(fn, arr)
                    acc[name].append(sz / dt / 1000)
            row = [f'{dname}/{sz:>5}']
            for name in variants:
                row.append(f'{sum(acc[name])/len(acc[name]):>9.1f}')
            print(' '.join(row))
        print()

if __name__ == '__main__':
    main()

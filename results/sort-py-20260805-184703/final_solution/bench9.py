import random, time, bisect, heapq
from itertools import chain

def radix(arr, base_small=256, base_large=2048, large_n=8000):
    n = len(arr)
    if n < 2:
        return list(arr)
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

def make_bisect(bisect_n, cnt_mult=4):
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
        return radix(arr, 256, 2048, 8000)
    return v

def make_mergeheap(bisect_n, k, stride=True, cnt_mult=4, radix_large_n=8000):
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
        if n <= 6000:
            if stride:
                chunks = [v(arr[i::k]) for i in range(k)]
            else:
                step = (n + k - 1) // k
                chunks = [v(arr[i*step:(i+1)*step]) for i in range(k)]
            return list(heapq.merge(*chunks))
        return radix(arr, 256, 2048, radix_large_n)
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

def bench(fn, arr, reps=7):
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
        'bisect1024': make_bisect(1024),
        'bisect2000': make_bisect(2000),
        'bisect3000': make_bisect(3000),
        'm4':  make_mergeheap(1024, 4, True),
        'm8':  make_mergeheap(1024, 8, True),
        'm16': make_mergeheap(1024, 16, True),
        'm8c': make_mergeheap(1024, 8, False),
        'm32': make_mergeheap(1024, 32, True),
    }
    fns = variants
    for name, fn in fns.items():
        assert check(fn), name

    sizes = [500, 1000, 1500, 2000, 2500, 3000, 4000, 6000, 8000, 12000]
    dists = {
        'r30': lambda rnd, sz: [rnd.randint(-(2**30), 2**30) for _ in range(sz)],
        'r63': lambda rnd, sz: [rnd.randint(-(2**63), 2**63) for _ in range(sz)],
        'small': lambda rnd, sz: [rnd.randint(-1000, 1000) for _ in range(sz)],
    }
    for dname, gen in dists.items():
        print('=== dist', dname)
        print('size      ' + ' '.join(f'{n:>9}' for n in fns))
        for sz in sizes:
            acc = {n: [] for n in fns}
            for seed in range(3):
                rnd = random.Random(300 + seed)
                arr = gen(rnd, sz)
                for name, fn in fns.items():
                    dt = bench(fn, arr)
                    acc[name].append(sz / dt / 1000)
            row = [f'{sz:>5}']
            for name in fns:
                row.append(f'{sum(acc[name])/len(acc[name]):>9.1f}')
            print(' '.join(row))

if __name__ == '__main__':
    main()

import random, time, bisect
from itertools import chain

def make_v(bisect_n, cnt_mult, base=256):
    mask = base - 1
    nb = base.bit_length() - 1
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
        if mn < 0:
            a = [x - mn for x in arr]; mx -= mn
        else:
            a = list(arr)
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
    for trial in range(400):
        n = rnd.randint(0, 100)
        lo = rnd.choice([-5, -100, 0, 10, -10**9])
        hi = rnd.choice([5, 100, 1000, 10**9, 10**9])
        if hi < lo:
            lo, hi = hi, lo
        arr = [rnd.randint(lo, hi) for _ in range(n)]
        if fn(arr) != sorted(arr):
            return False
    for trial in range(80):
        n = rnd.randint(100, 5000)
        arr = [rnd.randint(-(2**60), 2**60) for _ in range(n)]
        if fn(arr) != sorted(arr):
            return False
    for trial in range(80):
        n = rnd.randint(100, 5000)
        arr = [rnd.randint(0, 50) for _ in range(n)]
        if fn(arr) != sorted(arr):
            return False
    return True

def bench(fn, arr, reps=1):
    fn(arr)
    t0 = time.perf_counter()
    for _ in range(reps):
        fn(arr)
    return (time.perf_counter() - t0) / reps

def geomean(xs):
    p = 1.0
    for x in xs:
        p *= x
    return p ** (1.0 / len(xs))

def main():
    configs = {
        'cur': make_v(32, 2),
        'b128c4': make_v(128, 4),
        'b192c4': make_v(192, 4),
        'b256c4': make_v(256, 4),
        'b384c4': make_v(384, 4),
        'b512c4': make_v(512, 4),
        'b128c8': make_v(128, 8),
        'b256c8': make_v(256, 8),
    }
    for name, fn in configs.items():
        assert check(fn), name

    sizes = [10, 30, 100, 300, 500, 1000, 3000, 10000, 30000, 100000, 300000, 1000000]
    dists = {
        'r30': lambda rnd, sz: [rnd.randint(-(2**30), 2**30) for _ in range(sz)],
        'r64': lambda rnd, sz: [rnd.randint(-(2**63), 2**63) for _ in range(sz)],
        'small': lambda rnd, sz: [rnd.randint(-1000, 1000) for _ in range(sz)],
        'med': lambda rnd, sz: [rnd.randint(-(10**6), 10**6) for _ in range(sz)],
    }
    # average over 3 seeds per dist
    agg = {d: {n: [] for n in configs} for d in dists}
    for seed in range(3):
        rnd = random.Random(100 + seed)
        for dname, gen in dists.items():
            for sz in sizes:
                arr = gen(rnd, sz)
                for name, fn in configs.items():
                    dt = bench(fn, arr)
                    agg[dname][name].append(sz / dt / 1000)
    print('dist    ' + ' '.join(f'{n:>8}' for n in configs))
    for dname in dists:
        gms = {}
        row = [f'{dname:>7}']
        for name in configs:
            gms[name] = geomean(agg[dname][name])
            row.append(f'{gms[name]:>8.1f}')
        print(' '.join(row))

if __name__ == '__main__':
    main()

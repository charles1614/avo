import random, time, bisect
from itertools import chain

def make_v(bisect_n, cnt_mult, base):
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
    for trial in range(300):
        n = rnd.randint(0, 120)
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
    return True

def bench(fn, arr, reps=2):
    fn(arr)
    t0 = time.perf_counter()
    for _ in range(reps):
        fn(arr)
    return (time.perf_counter() - t0) / reps

def main():
    configs = {
        'b1024c4_256': make_v(1024, 4, 256),
        'b1024c4_1024': make_v(1024, 4, 1024),
        'b1024c4_2048': make_v(1024, 4, 2048),
    }
    for name, fn in configs.items():
        assert check(fn), name
    sizes = [1024, 5000, 10000, 30000, 100000, 300000, 1000000]
    dists = {'r30': lambda rnd, sz: [rnd.randint(-(2**30), 2**30) for _ in range(sz)],
             'r64': lambda rnd, sz: [rnd.randint(-(2**63), 2**63) for _ in range(sz)]}
    print('dist/size ' + ' '.join(f'{n:>8}' for n in configs))
    for dname, gen in dists.items():
        for sz in sizes:
            acc = {n: [] for n in configs}
            for seed in range(5):
                rnd = random.Random(300 + seed)
                arr = gen(rnd, sz)
                for name, fn in configs.items():
                    acc[name].append(sz / bench(fn, arr) / 1000)
            row = [f'{dname}/{sz:>4}']
            for name in configs:
                row.append(f'{sum(acc[name])/len(acc[name]):>8.1f}')
            print(' '.join(row))

if __name__ == '__main__':
    main()

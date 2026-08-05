import random, time, bisect, heapq
from itertools import chain

def make_radix(base_small=256, base_large=2048, large_n=8000, cnt_mult=4, merge_max=0, k=8):
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
        if r <= cnt_mult * n:
            cnt = [0] * r
            for x in arr:
                cnt[x - mn] += 1
            out = []
            for i, c in enumerate(cnt):
                if c:
                    out.extend([mn + i] * c)
            return out
        if merge_max and n <= merge_max:
            chunks = [v(arr[i::k]) for i in range(k)]
            return list(heapq.merge(*chunks))
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
        t0 = time.perf_counter(); fn(arr); ts.append(time.perf_counter()-t0)
    ts.sort()
    return ts[len(ts)//2]

def main():
    variants = {
        'b256': make_radix(256, 256),
        'b512': make_radix(512, 512),
        'b1024': make_radix(1024, 1024),
        'b2048': make_radix(2048, 2048),
        'b4096': make_radix(4096, 4096),
        'b8192': make_radix(8192, 8192),
        'b256/2048': make_radix(256, 2048, 8000),
        'merge8@8000': make_radix(256, 2048, 8000, merge_max=8000),
        'merge8@6000': make_radix(256, 2048, 8000, merge_max=6000),
        'merge8@4000': make_radix(256, 2048, 8000, merge_max=4000),
        'merge16@8000': make_radix(256, 2048, 8000, merge_max=8000, k=16),
    }
    for name, fn in variants.items():
        assert check(fn), name

    sizes = [8000]
    ranges = [2**19, 2**20, 2**21, 2**22, 2**23, 2**30]
    for rng in ranges:
        print('=== range', rng)
        print('size      ' + ' '.join(f'{n:>9}' for n in variants))
        for sz in sizes:
            acc = {n: [] for n in variants}
            for seed in range(5):
                rnd = random.Random(700 + seed)
                arr = [rnd.randint(-rng, rng) for _ in range(sz)]
                for name, fn in variants.items():
                    acc[name].append(sz / bench(fn, arr) / 1000)
            row = [f'{sz:>5}']
            for name in variants:
                row.append(f'{sum(acc[name])/len(acc[name]):>9.1f}')
            print(' '.join(row))

if __name__ == '__main__':
    main()

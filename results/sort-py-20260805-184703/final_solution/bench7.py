import random, time, bisect
from itertools import chain

def make_v(bisect_n, cnt_mult, base_small=256, base_large=2048, large_n=8000,
           use_count_radix=False, B=65536, count_radix_n=16000):
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
        if use_count_radix and n >= count_radix_n:
            mask = B - 1
            nb = B.bit_length() - 1
            shift = 0
            while mx >> shift:
                cnt = [0] * B
                m = mask
                for x in a:
                    cnt[(x >> shift) & m] += 1
                s = 0
                for i in range(B):
                    c = cnt[i]
                    cnt[i] = s
                    s += c
                out = [0] * n
                for x in a:
                    k = (x >> shift) & m
                    out[cnt[k]] = x
                    cnt[k] += 1
                a = out
                shift += nb
        else:
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

def bench(fn, arr, reps=5):
    fn(arr)
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn(arr)
        ts.append(time.perf_counter() - t0)
    ts.sort()
    return ts[len(ts)//2]

def geomean(xs):
    p = 1.0
    for x in xs:
        p *= x
    return p ** (1.0 / len(xs))

def main():
    configs = {
        'cur': make_v(1024, 4, 256, 2048, 8000),
        'b1024_256': make_v(1024, 4, 256, 256, 8000),
        'b1024_1024': make_v(1024, 4, 1024, 1024, 8000),
        'b1024_2048': make_v(1024, 4, 1024, 2048, 8000),
        'b1024_4096': make_v(1024, 4, 1024, 4096, 8000),
        'b1024_8192': make_v(1024, 4, 1024, 8192, 8000),
        'b768_2048': make_v(768, 4, 256, 2048, 8000),
        'b1280_2048': make_v(1280, 4, 256, 2048, 8000),
        'b1536_2048': make_v(1536, 4, 256, 2048, 8000),
        'cnt65536@8k': make_v(1024, 4, 256, 2048, 8000, True, 65536, 8000),
        'cnt65536@16k': make_v(1024, 4, 256, 2048, 8000, True, 65536, 16000),
        'cnt4096@8k': make_v(1024, 4, 256, 2048, 8000, True, 4096, 8000),
    }
    for name, fn in configs.items():
        assert check(fn), name

    sizes = [500, 2000, 8000]
    dists = {
        'r30': lambda rnd, sz: [rnd.randint(-(2**30), 2**30) for _ in range(sz)],
        'r31': lambda rnd, sz: [rnd.randint(-(2**31), 2**31) for _ in range(sz)],
        'r63': lambda rnd, sz: [rnd.randint(-(2**63), 2**63) for _ in range(sz)],
        'small': lambda rnd, sz: [rnd.randint(-1000, 1000) for _ in range(sz)],
        'dup': lambda rnd, sz: [rnd.randint(0, 50) for _ in range(sz)],
    }
    for dname, gen in dists.items():
        print('=== dist', dname)
        print('size      ' + ' '.join(f'{n:>9}' for n in configs))
        gms = {n: [] for n in configs}
        for sz in sizes:
            acc = {n: [] for n in configs}
            for seed in range(3):
                rnd = random.Random(200 + seed)
                arr = gen(rnd, sz)
                for name, fn in configs.items():
                    dt = bench(fn, arr)
                    acc[name].append(sz / dt / 1000)
            row = [f'{sz:>5}']
            for name in configs:
                m = sum(acc[name])/len(acc[name])
                gms[name].append(m)
                row.append(f'{m:>9.1f}')
            print(' '.join(row))
        print('geo-mean  ' + ' '.join(f'{geomean(gms[n]):>9.1f}' for n in configs))

if __name__ == '__main__':
    main()

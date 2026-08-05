import random, time, bisect
from itertools import chain

def make_v(bisect_n=96, cnt_mult=8, base_small=256, base_med=1024, base_large=2048,
           med_n=2000, large_n=20000, ins_n=0):
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
        if n < med_n:
            bsz, mask, nb = base_small, base_small - 1, base_small.bit_length() - 1
        elif n < large_n:
            bsz, mask, nb = base_med, base_med - 1, base_med.bit_length() - 1
        else:
            bsz, mask, nb = base_large, base_large - 1, base_large.bit_length() - 1
        shift = 0
        while mx >> shift:
            buckets = [[] for _ in range(bsz)]
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
    # duplicates-heavy
    for trial in range(80):
        n = rnd.randint(100, 5000)
        arr = [rnd.randint(0, 50) for _ in range(n)]
        if fn(arr) != sorted(arr):
            return False
    return True

def bench(fn, arr):
    fn(arr)
    t0 = time.perf_counter()
    fn(arr)
    return time.perf_counter() - t0

def geomean(xs):
    p = 1.0
    for x in xs:
        p *= x
    return p ** (1.0 / len(xs))

def main():
    configs = {
        'cur(32ins,2n,256)': make_v(bisect_n=32, cnt_mult=2, base_small=256, base_med=256, base_large=256),
        'b64':  make_v(bisect_n=64, cnt_mult=2),
        'b96':  make_v(bisect_n=96, cnt_mult=2),
        'b128': make_v(bisect_n=128, cnt_mult=2),
        'b192': make_v(bisect_n=192, cnt_mult=2),
        'b256': make_v(bisect_n=256, cnt_mult=2),
        'b96c4': make_v(bisect_n=96, cnt_mult=4),
        'b96c16': make_v(bisect_n=96, cnt_mult=16),
        'b96c32': make_v(bisect_n=96, cnt_mult=32),
    }
    for name, fn in configs.items():
        assert check(fn), name

    rnd = random.Random(7)
    sizes = [10, 30, 100, 300, 1000, 3000, 10000, 30000, 100000, 300000, 1000000]
    dists = {
        'r30': lambda sz: [rnd.randint(-(2**30), 2**30) for _ in range(sz)],
        'r64': lambda sz: [rnd.randint(-(2**63), 2**63) for _ in range(sz)],
        'small': lambda sz: [rnd.randint(-1000, 1000) for _ in range(sz)],
        'dup': lambda sz: [rnd.randint(0, max(2, sz // 10)) for _ in range(sz)],
    }
    for dname, gen in dists.items():
        print('=== dist', dname)
        print('size      ' + ' '.join(f'{n:>8}' for n in configs))
        gms = {n: [] for n in configs}
        for sz in sizes:
            arr = gen(sz)
            row = [f'{sz:>5}']
            for name, fn in configs.items():
                dt = bench(fn, arr)
                kps = sz / dt / 1000
                row.append(f'{kps:>8.1f}')
                gms[name].append(kps)
            print(' '.join(row))
        print('geo-mean  ' + ' '.join(f'{geomean(gms[n]):>8.1f}' for n in configs))

if __name__ == '__main__':
    main()

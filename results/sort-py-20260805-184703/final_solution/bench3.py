import random, time, bisect
from itertools import chain

def make_bucket(bisect_n, cnt_mult, bases):
    # bases: (n_low, base) pairs; last applies to all larger
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
            bsz = 256
            for lo, b in bases:
                if n < lo:
                    break
                bsz = b
            mask = bsz - 1
            nb = bsz.bit_length() - 1
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

def make_count_radix(bisect_n, cnt_mult, radix_n_min, B, nb):
    mask = B - 1
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
        if n < radix_n_min:
            # bucket style base 256
            shift = 0
            while mx >> shift:
                buckets = [[] for _ in range(256)]
                for x in a:
                    buckets[(x >> shift) & 255].append(x)
                a = list(chain.from_iterable(buckets))
                shift += 8
        else:
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
        'bucket(128,4n,adapt)': make_bucket(128, 4, [(2000, 256), (20000, 1024), (1e18, 2048)]),
        'cntB(128,4n,16k,65536)': make_count_radix(128, 4, 16000, 65536, 16),
        'cntB(128,4n,8k,65536)': make_count_radix(128, 4, 8000, 65536, 16),
        'cntB(128,4n,32k,65536)': make_count_radix(128, 4, 32000, 65536, 16),
        'cntB(128,4n,16k,4096)': make_count_radix(128, 4, 16000, 4096, 12),
        'cntB(128,4n,16k,16384)': make_count_radix(128, 4, 16000, 16384, 14),
        'bucket(128,4n,256all)': make_bucket(128, 4, [(1e18, 256)]),
    }
    for name, fn in configs.items():
        assert check(fn), name

    rnd = random.Random(7)
    sizes = [10, 100, 1000, 10000, 100000, 1000000]
    dists = {
        'r30': lambda sz: [rnd.randint(-(2**30), 2**30) for _ in range(sz)],
        'r64': lambda sz: [rnd.randint(-(2**63), 2**63) for _ in range(sz)],
        'small': lambda sz: [rnd.randint(-1000, 1000) for _ in range(sz)],
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

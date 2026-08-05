import random, time, bisect, heapq
from itertools import chain

def v_bucket(arr, bisect_n=1024, cnt_mult=4, base_small=256, base_large=2048, large_n=8000):
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

def v_countradix(arr, bisect_n=1024, cnt_mult=4, base=256):
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
    mask = base - 1
    nb = base.bit_length() - 1
    shift = 0
    while mx >> shift:
        cnt = [0] * base
        m = mask
        for x in a:
            cnt[(x >> shift) & m] += 1
        s = 0
        for i in range(base):
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

def v_bisect_all(arr, bisect_n=10**9, cnt_mult=4):
    # pure bisect for everything (no counting/radix)
    n = len(arr)
    if n < 2:
        return list(arr)
    out = []
    ins = bisect.insort
    for x in arr:
        ins(out, x)
    return out

def v_merge2(arr):
    # split, bisect-sort halves, merge
    n = len(arr)
    if n < 2:
        return list(arr)
    half = n // 2
    a = v_bisect_all(arr[:half])
    b = v_bisect_all(arr[half:])
    out = []
    i = j = 0
    la, lb = len(a), len(b)
    while i < la and j < lb:
        if a[i] <= b[j]:
            out.append(a[i]); i += 1
        else:
            out.append(b[j]); j += 1
    out.extend(a[i:])
    out.extend(b[j:])
    return out

def v_merge8_heap(arr):
    n = len(arr)
    if n < 2:
        return list(arr)
    k = 8
    chunks = [v_bisect_all(arr[i::k]) for i in range(k)]
    return list(heapq.merge(*chunks))

def v_merge8(arr):
    n = len(arr)
    if n < 2:
        return list(arr)
    k = 8
    chunks = [v_bisect_all(arr[i::k]) for i in range(k)]
    while len(chunks) > 1:
        new = []
        for i in range(0, len(chunks), 2):
            if i + 1 < len(chunks):
                new.append(_merge(chunks[i], chunks[i+1]))
            else:
                new.append(chunks[i])
        chunks = new
    return chunks[0]

def _merge(a, b):
    out = []
    i = j = 0
    la, lb = len(a), len(b)
    while i < la and j < lb:
        if a[i] <= b[j]:
            out.append(a[i]); i += 1
        else:
            out.append(b[j]); j += 1
    out.extend(a[i:])
    out.extend(b[j:])
    return out

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
        'cur': lambda: v_bucket,
        'cnt256': lambda: v_countradix,
        'cnt512': lambda: (lambda arr: v_countradix(arr, base=512)),
        'cnt1024': lambda: (lambda arr: v_countradix(arr, base=1024)),
        'bisect': lambda: v_bisect_all,
        'merge2': lambda: v_merge2,
        'merge8': lambda: v_merge8,
        'm8heap': lambda: v_merge8_heap,
        'radix512': lambda: (lambda arr: v_bucket(arr, base_small=512, base_large=2048)),
        'radix1024': lambda: (lambda arr: v_bucket(arr, base_small=1024, base_large=2048)),
    }
    fns = {k: f() for k, f in variants.items()}
    for name, fn in fns.items():
        assert check(fn), name

    sizes = [500, 2000, 8000]
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

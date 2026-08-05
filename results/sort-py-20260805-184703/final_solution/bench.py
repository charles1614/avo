import random, time, sys, bisect
from itertools import chain

def v0(arr):
    n = len(arr)
    if n < 2:
        return list(arr)
    if n <= 32:
        out = list(arr)
        for i in range(1, n):
            v = out[i]
            j = i - 1
            while j >= 0 and out[j] > v:
                out[j + 1] = out[j]
                j -= 1
            out[j + 1] = v
        return out
    mn = min(arr); mx = max(arr)
    r = mx - mn + 1
    if r <= 2 * n:
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
        buckets = [[] for _ in range(256)]
        for x in a:
            buckets[(x >> shift) & 255].append(x)
        a = [x for b in buckets for x in b]
        shift += 8
    if mn < 0:
        return [x + mn for x in a]
    return a

def v1(arr):  # chain flatten
    n = len(arr)
    if n < 2:
        return list(arr)
    if n <= 32:
        out = list(arr)
        for i in range(1, n):
            v = out[i]
            j = i - 1
            while j >= 0 and out[j] > v:
                out[j + 1] = out[j]
                j -= 1
            out[j + 1] = v
        return out
    mn = min(arr); mx = max(arr)
    r = mx - mn + 1
    if r <= 2 * n:
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
        buckets = [[] for _ in range(256)]
        for x in a:
            buckets[(x >> shift) & 255].append(x)
        a = list(chain.from_iterable(buckets))
        shift += 8
    if mn < 0:
        return [x + mn for x in a]
    return a

def v2(arr, base=1024):  # adaptive base + chain
    n = len(arr)
    if n < 2:
        return list(arr)
    if n <= 32:
        out = list(arr)
        for i in range(1, n):
            v = out[i]
            j = i - 1
            while j >= 0 and out[j] > v:
                out[j + 1] = out[j]
                j -= 1
            out[j + 1] = v
        return out
    mn = min(arr); mx = max(arr)
    r = mx - mn + 1
    if r <= 2 * n:
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
    if n < 2000:
        bsz, mask = 256, 255
    elif n < 20000:
        bsz, mask = 1024, 1023
    else:
        bsz, mask = 2048, 2047
    nb = bsz.bit_length() - 1
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

def v3(arr):  # bisect insort for small n
    n = len(arr)
    if n < 2:
        return list(arr)
    if n <= 128:
        out = []
        ins = bisect.insort
        for x in arr:
            ins(out, x)
        return out
    mn = min(arr); mx = max(arr)
    r = mx - mn + 1
    if r <= 2 * n:
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
        buckets = [[] for _ in range(256)]
        for x in a:
            buckets[(x >> shift) & 255].append(x)
        a = list(chain.from_iterable(buckets))
        shift += 8
    if mn < 0:
        return [x + mn for x in a]
    return a

def v4(arr):  # counting threshold 8n + chain
    n = len(arr)
    if n < 2:
        return list(arr)
    if n <= 32:
        out = list(arr)
        for i in range(1, n):
            v = out[i]
            j = i - 1
            while j >= 0 and out[j] > v:
                out[j + 1] = out[j]
                j -= 1
            out[j + 1] = v
        return out
    mn = min(arr); mx = max(arr)
    r = mx - mn + 1
    if r <= 8 * n:
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
        buckets = [[] for _ in range(256)]
        for x in a:
            buckets[(x >> shift) & 255].append(x)
        a = list(chain.from_iterable(buckets))
        shift += 8
    if mn < 0:
        return [x + mn for x in a]
    return a

def v5(arr):  # everything: bisect small, threshold 8n, adaptive base, chain
    n = len(arr)
    if n < 2:
        return list(arr)
    if n <= 96:
        out = []
        ins = bisect.insort
        for x in arr:
            ins(out, x)
        return out
    mn = min(arr); mx = max(arr)
    r = mx - mn + 1
    if r <= 8 * n:
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
    if n < 2000:
        bsz, mask, nb = 256, 255, 8
    elif n < 20000:
        bsz, mask, nb = 1024, 1023, 10
    else:
        bsz, mask, nb = 2048, 2047, 11
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

VARIANTS = {'v0': v0, 'v1': v1, 'v2': v2, 'v3': v3, 'v4': v4, 'v5': v5}

def check(name, fn):
    rnd = random.Random(42)
    for trial in range(200):
        n = rnd.randint(0, 80)
        lo = rnd.choice([-5, -100, 0, 10])
        hi = rnd.choice([5, 100, 1000, 10**9])
        if hi < lo:
            lo, hi = hi, lo
        arr = [rnd.randint(lo, hi) for _ in range(n)]
        ref = sorted(arr)
        got = fn(arr)
        if got != ref:
            print(f'{name} FAIL n={n} arr={arr[:20]} got={got[:20]}')
            return False
    for trial in range(50):
        n = rnd.randint(100, 3000)
        arr = [rnd.randint(-(2**40), 2**40) for _ in range(n)]
        if fn(arr) != sorted(arr):
            print(f'{name} FAIL big n={n}')
            return False
    return True

def bench(fn, arr):
    fn(arr)  # warmup
    t0 = time.perf_counter()
    fn(arr)
    t1 = time.perf_counter()
    return t1 - t0

def main():
    for name, fn in VARIANTS.items():
        if not check(name, fn):
            print('SKIP', name)
    rnd = random.Random(7)
    sizes = [10, 30, 100, 300, 1000, 3000, 10000, 30000, 100000, 300000, 1000000]
    print('size      ' + ' '.join(f'{n:>9}' for n in VARIANTS))
    for sz in sizes:
        arr = [rnd.randint(-(2**30), 2**30) for _ in range(sz)]
        row = [f'{sz:>5}']
        for name, fn in VARIANTS.items():
            dt = bench(fn, arr)
            kps = sz / dt / 1000
            row.append(f'{kps:>9.1f}')
        print(' '.join(row))

if __name__ == '__main__':
    main()

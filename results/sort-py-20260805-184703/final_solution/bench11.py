import random, time, bisect
from itertools import chain

def radix(arr, base=256, cnt_mult=4):
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
    if mn < 0:
        a = [x - mn for x in arr]; mx -= mn
    else:
        a = list(arr)
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

def bench(fn, arr, reps=9):
    fn(arr)
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); fn(arr); ts.append(time.perf_counter()-t0)
    ts.sort()
    return ts[len(ts)//2]

def main():
    n = 2000
    ranges = [2**16, 2**18, 2**20, 2**21, 2**22, 2**24, 2**26, 2**28, 2**31]
    bases = [256, 1024, 2048, 4096]
    print('range     ' + ' '.join(f'b{b:<7}' for b in bases))
    for rng in ranges:
        acc = {b: [] for b in bases}
        for seed in range(5):
            rnd = random.Random(500 + seed)
            arr = [rnd.randint(-rng, rng) for _ in range(n)]
            for b in bases:
                acc[b].append(n / bench(radix, arr, b) / 1000)
        row = [f'{rng:>9}']
        for b in bases:
            row.append(f'{sum(acc[b])/len(acc[b]):>8.1f}')
        print(' '.join(row))

if __name__ == '__main__':
    main()

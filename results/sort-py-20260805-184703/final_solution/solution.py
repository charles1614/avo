"""Pure-Python integer sort.

Hybrid strategy:
  * tiny/mid lists  -> bisect.insort into a fresh list (C-speed binary
                       search + list.insert; beats radix setup up to ~1k)
  * narrow range    -> counting sort (linear in n + range)
  * otherwise       -> LSD radix sort, bucket-append style with
                       C-speed flatten via itertools.chain; base 256 for
                       mid lists, base 2048 for large lists (fewer passes)

The input list is never mutated; a new sorted list is returned.
"""

import bisect
from itertools import chain


def sort_list(arr):
    n = len(arr)
    if n < 2:
        return list(arr)

    # Small-to-mid lists: bisect + insert beats radix/counting setup cost.
    if n <= 1024:
        out = []
        ins = bisect.insort
        for x in arr:
            ins(out, x)
        return out

    mn = min(arr)
    mx = max(arr)
    r = mx - mn + 1

    # Narrow range (or many duplicates): counting sort is linear.
    if r <= 4 * n:
        cnt = [0] * r
        for x in arr:
            cnt[x - mn] += 1
        out = []
        for i, c in enumerate(cnt):
            if c:
                out.extend([mn + i] * c)
        return out

    # Wide range: LSD radix sort, bucket style.
    if mn < 0:
        a = [x - mn for x in arr]
        mx -= mn
    else:
        a = list(arr)
    if n >= 8000:
        base, mask, nb = 2048, 2047, 11
    else:
        base, mask, nb = 256, 255, 8
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

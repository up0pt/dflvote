import math
from typing import List


def generate_divisions(n: int) -> List[List[int]]:
    """
    Generate partitions for indices 0..n-1, where n is a power of two.
    Partitions are ordered by descending group count: n, n/2, ..., 2, 1.
    For each group count g = 2^k:
      - Always add contiguous grouping: split into g blocks of size n/g.
      - If g > 2, also add interleaved grouping: compute half = (n/g)//2,
        then group j is [i for i in range(n) if (i//half) % g == j].
    Returns a list of partitions, each a list of groups (lists of indices).
    """
    assert n > 0 and (n & (n - 1)) == 0, "n must be a power of two"
    splits: List[List[int]] = []
    m = int(math.log2(n))
    # iterate group counts from n down to 1
    for k in range(m, -1, -1):
        g = 2 ** k
        size = n // g
        # contiguous grouping
        cont = [list(range(j * size, (j + 1) * size)) for j in range(g)]
        splits.append(cont)
                # interleaved grouping for g >= 2 (size>1 ensures valid grouping)
        if g >= 2 and size > 1:
            half = size // 2
            inter = [[i for i in range(n) if (i // half) % g == j] for j in range(g)]
            splits.append(inter)
    return splits


if __name__ == "__main__":
    print(generate_divisions(32))
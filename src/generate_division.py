import math
import random

def generate_divisions(n: int) -> list[list[list[int]]]:
    """
    Generate partitions for indices 0..n-1, where n is a power of two.
    Partitions are ordered by descending group count: n, n/2, ..., 2, 1.
    For each group count g = 2^k:
      - Always add contiguous grouping: split into g blocks of size n/g.
      - If g > 2, also add stride interleaved grouping: group j is
        [j + k * g for k in range(n // g)]
    Returns a list of partitions, each a list of groups (lists of indices).
    """
    assert n > 0 and (n & (n - 1)) == 0, "n must be a power of two"
    splits = []
    m = int(math.log2(n))
    for k in range(m, -1, -1):
        g = 2 ** k
        size = n // g
        # contiguous grouping
        cont = [list(range(j * size, (j + 1) * size)) for j in range(g)]
        splits.append(cont)
        # stride-based interleaved grouping (e.g., [0, 8, 16, 24])
        if g >= 2 and g < n:
            inter = [[j + k * g for k in range(size)] for j in range(g)]
            splits.append(inter)
    return splits

def random_dividions(n: int) -> list[list[int]]:
    """
    Generate random partitions for indices 0..n-1, where n is a power of two.
    Partitions are ordered by descending group count: n, n/2, ..., 2, 1.
    For each group count g = 2^k, randomly assign indices to g groups.
    Returns a list of partitions, each a list of groups (lists of indices).
    """
    assert n > 0 and (n & (n - 1)) == 0, "n must be a power of two"
    indices = list(range(n))
    partitions = []
    m = int(math.log2(n))
    for k in range(m, -1, -1):
        g = 2 ** k
        size = n // g
        shuffled = indices[:]
        random.shuffle(shuffled)
        groups = [shuffled[j * size:(j + 1) * size] for j in range(g)]
        partitions.append(groups)
    return partitions
    


if __name__ == "__main__":
    a = generate_divisions(32)
    print(a)
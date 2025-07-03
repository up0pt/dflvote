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

def random_div_only(n: int, m: int, rec: int) -> list[list[list[int]]]:
    """
    0 から n-1 までの整数を、サイズ m のグループに分割し、重複しないように rec 個の異なるランダムな組み合わせを生成する。

    Args:
        n (int): 全体の要素数（0 から n-1 まで）。
        m (int): 1 グループあたりの要素数。
        rec (int): 生成する組み合わせの数。

    Returns:
        List[List[List[int]]]: rec 個分の組み合わせ。各組み合わせはグループのリストで、その各要素は整数のリスト。

    Raises:
        ValueError: n が m の倍数でない場合。
    """
    if n <= 0 or m <= 0 or rec <= 0:
        raise ValueError("n, m, rec はすべて正の整数で指定してください")
    if n % m != 0:
        raise ValueError(f"n={n} は m={m} の倍数ではありません。グループを等分できません。")

    random.seed(42)
    partitions: list[list[list[int]]] = []
    for _ in range(rec):
        elements = list(range(n))
        random.shuffle(elements)
        # グループに分割
        groups = [elements[i : i + m] for i in range(0, n, m)]
        partitions.append(groups)
    return partitions

if __name__ == "__main__":
    print(random_div_only(12, 3, 2))
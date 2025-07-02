import numpy as np

def generate_dirichlet_dist(num_clients: int, dist_groups: list[list[int]], alphas: list[float]) -> list[list[float]]:
    validate_dist_groups(num_clients, dist_groups)
    
    # 各グループに対して1つのdirichlet値を生成
    group_weights = [np.random.dirichlet(alphas) for _ in range(len(dist_groups))]

    # 各値をマッピングする辞書を作成
    value_to_weight = {}
    for group, weight in zip(dist_groups, group_weights):
        for val in group:
            value_to_weight[val] = weight

    # リストを作成
    result = [value_to_weight[i] for i in range(num_clients)]
    return result

def validate_dist_groups(num_clients: int, dist_groups: list[list[int]]) -> bool:
    # フラットなリストに変換
    all_clients = [client for group in dist_groups for client in group]
    
    # ソートして期待されるリストと比較
    expected_clients = list(range(num_clients))
    
    # 条件確認：要素数・重複なし・範囲内
    if sorted(all_clients) != expected_clients:
        raise ValueError("Distribution is wrong: any missing values or duplicated values")

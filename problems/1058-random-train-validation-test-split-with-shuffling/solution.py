import numpy as np

def random_split(
    data: np.ndarray,
    train_frac: float,
    validation_frac: float,
    seed: int = 123
) -> list:
    """
    Randomly split a dataset into train, validation, and test subsets.
    """

    # 1. 检查输入比例是否合法
    if not 0 <= train_frac <= 1:
        raise ValueError("train_frac must be between 0 and 1.")
    if not 0 <= validation_frac <= 1:
        raise ValueError("validation_frac must be between 0 and 1.")
    if train_frac + validation_frac > 1:
        raise ValueError("train_frac + validation_frac must not exceed 1.")

    # 2. 创建随机数生成器
    # seed 固定后，每次运行都会得到相同的随机划分结果，方便复现
    rng = np.random.default_rng(seed)

    # 3. 随机打乱数据的索引
    # 比如原来的索引是 [0, 1, 2, 3, 4]
    # 可能会被打乱成 [4, 1, 3, 0, 2]
    indices = rng.permutation(len(data))

    # 4. 计算训练集和验证集需要多少条数据

    n_train = int(len(data) * train_frac)
    n_validation = int(len(data) * validation_frac)
    # 5. 从随机索引中切出 train / validation / test
    train_idx = indices[:n_train]

    validation_idx = indices[
        n_train:n_train + n_validation
    ]

    # 剩下的数据全部作为 test set
    test_idx = indices[
        n_train + n_validation:
    ]

    # 6. 根据索引从原数据中取出对应的数据
    return [
        data[train_idx],
        data[validation_idx],
        data[test_idx]
    ]
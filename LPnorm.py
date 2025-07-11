# LPnorm.py
import numpy as np


def LPnorm(arr, p):
    """
    Lp范数归一化 (支持无穷大范数)

    参数:
        arr: 输入光谱数据 (n_samples, n_points)
        p: 范数阶数 (4, 10 或 np.inf)

    返回:
        归一化后的光谱
    """
    if p not in [4, 10, np.inf]:
        raise ValueError("p值必须是4、10或无穷大")

    arr = np.asarray(arr, dtype=np.float64)
    result = np.zeros_like(arr)

    for i in range(arr.shape[0]):
        if p == np.inf:
            # 无穷范数(最大绝对值归一化)
            norm = np.max(np.abs(arr[i]))
        else:
            # 普通Lp范数
            norm = np.linalg.norm(arr[i], ord=p)

        if norm > 1e-10:  # 防止除以零
            result[i] = arr[i] / norm
        else:
            result[i] = arr[i]  # 零向量保持不变

    return result

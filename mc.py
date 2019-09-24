import numpy as np


# P -> N
def p2n(p):
    # n_xy = p_xy + sum_{z ≠ y} p_xz (n_zy + 1)
    # n_xy - sum_{z ≠ y} p_xz n_zy = 1
    # Flatten n_xy by mapping (x, y) to x * d + y
    d = p.shape[0]
    a = np.eye(d * d)
    for x in range(d):
        for y in range(d):
            i = x * d + y
            for z in range(d):
                if z != y:
                    j = z * d + y
                    a[i, j] -= p[x, z]
    b = np.ones(d * d)
    # print(a)
    r = np.linalg.solve(a, b)
    return r.reshape((d, d))


# N -> P
def n2p(n):
    return n @ np.linalg.inv(n - np.diagflat(np.diagonal(n)) + 1)


if __name__ == '__main__':
    import math

    # Transition
    p0 = np.array([
        [0.9, 0.1],
        [0.01, 0.99],
    ])

    # First return times
    n0 = np.array([
        [0.9 * 1 + 0.1 * (100 + 1), 10],
        [100, 0.99 * 1 + 0.01 * (10 + 1)],
    ])

    print(np.abs(p2n(p0) - n0).max())
    print(np.abs(n2p(n0) - p0).max())

    # Random transitions
    # d = 5
    # p = np.random.dirichlet([1] * d, d)
    # Simple deterministic transitions
    p1 = np.array([
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 0],
    ])
    print(p2n(p1))
    print(np.abs(n2p(p2n(p1)) - p1).max())

    # Singular matrix => Some n_xy are inf
    # p = np.array([
    #     [0, 1, 0],
    #     [1, 0, 0],
    #     [1, 0, 0],
    # ])
    # n = np.array([
    #     [2, 1, math.inf],
    #     [1, 2, math.inf],
    #     [1, 2, math.inf],
    # ])
    # print(n2p(n))
    # p = np.array([
    #     [0, 1, 0],
    #     [1, 0, 0],
    #     [0, 0, 1],
    # ])
    # n = np.array([
    #     [2, 1, math.inf],
    #     [1, 2, math.inf],
    #     [math.inf, math.inf, 1],
    # ])
    # print(p2n(p))
    # print(n2p(n))

import numpy as np

def init_random_system(n, radius=35):
    x = np.random.randn(n, 3)
    r = radius * np.random.rand(n)**(1/3.)
    x /= np.sqrt(np.sum(x**2, axis=1))[:, None]
    x *= r[:, None]

    p = np.random.randn(n, 3)
    q = np.random.randn(n, 3)

    return x, p, q


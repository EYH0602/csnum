import pandas as pd
import jax.numpy as np
import matplotlib.pyplot as plt
from numerical_methods.root_finding_1_var \
    import Bisection, Fixed_Point, Newton, Secant


def f1(x):
    return np.sin(np.power((x - 1), 2)) / 2


def f2(x):
    return np.power(5, -1 * x) - 2


def plot_fn(f, r=(0, 1)):
    x = np.linspace(*r)
    y = f(x)
    plt.plot(x, np.zeros(np.shape(x)))
    plt.plot(x, y)


f1_root = pd.DataFrame(
    {
        "Bisection": Bisection(f1, (0.8, 1)),
        "Fixed_Point": \
            Fixed_Point(
                f1, 3, g=lambda x: np.sqrt(np.arcsin(0)) + 1
            ),
        "Newton": Newton(f1, 0.8),
        "Secant": Secant(f1, (0.8, 0.9)),
    }
)

print(f1_root)

f2_root = pd.DataFrame(
    {
        "Bisection": Bisection(f2, (-2, 2)),
        "Fixed_Point": Fixed_Point(f2, -0.43066),
        "Newton": Newton(f2, -2),
        "Secant": Secant(f2, (-2, -1)),
    }
)

print(f2_root)

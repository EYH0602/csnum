# %%
import pandas as pd
import jax.numpy as np
import matplotlib.pyplot as plt
from numerical_methods.root_finding_1_var import (
    Bisection,
    Fixed_Point,
    Newton,
    Secant,
)


def f1(x):
    return np.sin(np.power((x - 1), 2)) / 2


def f2(x):
    return np.power(5, -1 * x) - 2


def plot_fn(
    f, r=(0, 1), title="f(x)", xlabel="x", ylabel="f(x)"
):
    plt.figure()
    x = np.linspace(*r)
    y = f(x)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.plot(x, np.zeros(np.shape(x)))
    plt.plot(x, y)


def to_abs_err(df, real_root):
    return df.applymap(
        lambda x: np.abs(real_root - x)
    ).astype(np.float64)


# %%
plot_fn(f1, r=(0.8, 1.2), title="f_1(x)")
plt.savefig("./img/report1/f1.png")


# %%
f1_res = pd.DataFrame(
    {
        "Bisection": Bisection(f1, (0.8, 1)),
        "Fixed_Point": Fixed_Point(
            f1, 3, g=lambda x: np.sqrt(np.arcsin(0)) + 1
        ),
        "Newton": Newton(f1, 0.8),
        "Secant": Secant(f1, (0.8, 0.9)),
    }
)
f1_res


# %%
f1_err = to_abs_err(f1_res, 1)
f1_err


# %%


# %%
f1_err["iteration"] = list(range(15))
f1_err.plot(
    x="iteration",
    y=["Bisection", "Fixed_Point", "Newton", "Secant"],
    title="Comparison of Absolute Errors Approximating Root of f_1",
    legend=True,
    ylim=(0, 0.2),
    ylabel="Abs. Err",
)
plt.savefig("./img/report1/f1_err.png")


# %%
plot_fn(f2, r=(-2, 2), title="f_2(x)")
plt.savefig("./img/report1/f2.png")

# %%
f2_res = pd.DataFrame(
    {
        "Bisection": Bisection(f2, (-2, 2)),
        "Fixed_Point": Fixed_Point(f2, -0.43066),
        "Newton": Newton(f2, -2),
        "Secant": Secant(f2, (-2, -1)),
    }
)
f2_res


# %%
f2_err = to_abs_err(f2_res, -np.log(2) / np.log(5))


# %%
f2_err["iteration"] = list(range(15))
f2_err.plot(
    x="iteration",
    y=["Bisection", "Fixed_Point", "Newton", "Secant"],
    title="Comparison of Absolute Errors Approximating Root of f_2",
    legend=True,
    ylabel="Abs. Err",
)
plt.savefig("./img/report1/f2_err.png")



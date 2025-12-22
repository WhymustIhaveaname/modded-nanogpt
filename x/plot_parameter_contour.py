#! /usr/bin/env python3

import warnings
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import ray
from scipy.integrate import quad

from phi import nest_phi

warnings.filterwarnings("ignore")


DEFAULT_N = 5
A_MIN = 2.5
A_MAX = 4.5
B_MIN = -6.0
B_MAX = -4.0
GRID_SIZE = 200
REFINE_FACTOR = 2

A_OPT = 3.4445
B_OPT = -4.7750

ab_opt = None

ERROR_MODE = "sqMPv3"

FILE_SUFFIX_BASE = f"n{DEFAULT_N}_a{A_MIN}-{A_MAX}_b{B_MIN}-{B_MAX}"

if ERROR_MODE == "sq":
    ERROR_LABEL = rf"$\int_0^1 (\phi^{{{DEFAULT_N}}}(x) - 1)^2 dx$"
    FILE_SUFFIX = f"sq_{FILE_SUFFIX_BASE}"
    ab_opt = [
        (3.5772, -4.9120),
    ]

    def compute_error(a, b, c):
        def integrand(x):
            return (nest_phi(np.array([x]), DEFAULT_N, a, b, c)[0] - 1.0) ** 2

        result, _ = quad(integrand, 0.0, 1.0, limit=500)
        return result
elif ERROR_MODE in ("sqMP", "sqMPv2"):
    MP_N = 1280
    MP_UPPER = 2.0 / np.sqrt(MP_N)

    def rho(x):
        return np.sqrt(MP_N) * np.sqrt(4 - MP_N * x**2) / np.pi

    target = "1" if ERROR_MODE == "sqMP" else r"\bar{\phi}"
    ERROR_LABEL = rf"$\int_0^{{2/\sqrt{{n}}}} \rho(x) (\phi^{{{DEFAULT_N}}}(x) - {target})^2 dx$, $n={MP_N}$"
    FILE_SUFFIX = f"{ERROR_MODE}_n{MP_N}_{FILE_SUFFIX_BASE}"

    def compute_error(a, b, c):
        def phi(x):
            return nest_phi(np.array([x]), DEFAULT_N, a, b, c)[0]

        if ERROR_MODE == "sqMP":
            phi_ref = 1.0
        else:
            phi_ref, _ = quad(lambda x: rho(x) * phi(x), 0.0, MP_UPPER, limit=500)

        def integrand(x):
            return rho(x) * (phi(x) / phi_ref - 1.0) ** 2

        result, _ = quad(integrand, 0.0, MP_UPPER, limit=500)
        return result
elif ERROR_MODE == "sqMPv3":
    MP_M, MP_N = 768, 768 * 4
    if MP_M > MP_N:
        MP_M, MP_N = MP_N, MP_M
    MP_GAMMA = MP_M / MP_N
    MP_LAMBDA_MINUS = (1 - np.sqrt(MP_GAMMA)) ** 2
    MP_LAMBDA_PLUS = (1 + np.sqrt(MP_GAMMA)) ** 2
    MP_LOWER = (1 - np.sqrt(MP_GAMMA)) / np.sqrt(MP_N)
    MP_UPPER = (1 + np.sqrt(MP_GAMMA)) / np.sqrt(MP_N)

    def rho(x):
        nx2 = MP_N * x**2
        return np.sqrt((MP_LAMBDA_PLUS - nx2) * (nx2 - MP_LAMBDA_MINUS)) / (
            np.pi * MP_GAMMA * x
        )

    ERROR_LABEL = (
        rf"$\int \rho(x) (\phi^{{{DEFAULT_N}}}(x) - 1)^2 dx$, $m={MP_M}, n={MP_N}$"
    )
    FILE_SUFFIX = f"sqMPv3_m{MP_M}_n{MP_N}_{FILE_SUFFIX_BASE}"

    def compute_error(a, b, c):
        def phi(x):
            return nest_phi(np.array([x]), DEFAULT_N, a, b, c)[0]

        def integrand(x):
            return rho(x) * (phi(x) - 1.0) ** 2

        result, _ = quad(integrand, MP_LOWER, MP_UPPER, limit=500)
        return result
else:
    raise ValueError(f"Unknown ERROR_MODE: {ERROR_MODE}")


@ray.remote
def integrate_error(a: float, b: float) -> float:
    c = 0.701 - a - b
    try:
        result = compute_error(a, b, c)
        if np.isnan(result) or np.isinf(result):
            return np.nan
        if result >= 2.0:
            return 2.0
        return result
    except Exception:
        return np.nan


def find_ab_opt(A, B, Z_masked):
    global_idx = np.unravel_index(np.nanargmin(Z_masked), Z_masked.shape)
    global_opt = (A[global_idx], B[global_idx])

    mask = (
        (A >= A_OPT - 0.2)
        & (A <= A_OPT + 0.2)
        & (B >= B_OPT - 0.2)
        & (B <= B_OPT + 0.2)
    )
    Z_local = np.where(mask, Z_masked, np.inf)
    local_idx = np.unravel_index(np.nanargmin(Z_local), Z_local.shape)
    local_opt = (A[local_idx], B[local_idx])

    return [local_opt, global_opt]


def plot_parameter_contour() -> None:
    a_values = np.linspace(A_MIN, A_MAX, GRID_SIZE)
    b_values = np.linspace(B_MIN, B_MAX, GRID_SIZE)
    A, B = np.meshgrid(a_values, b_values)

    ray.init(ignore_reinit_error=True)

    Z = np.zeros_like(A)
    tasks = []
    task_indices = []

    for i, j in product(range(GRID_SIZE), range(GRID_SIZE)):
        tasks.append(integrate_error.remote(A[i, j], B[i, j]))
        task_indices.append((i, j))

    results = ray.get(tasks)
    for (i, j), result in zip(task_indices, results):
        Z[i, j] = result
    print(f"积分统计: 计算了 {len(tasks)} 个点")

    Z_masked = np.ma.masked_invalid(Z)

    # 对 Z < threshold 的格子进行加密
    refine_threshold = abs(Z_masked.min()) * 2
    low_mask = Z_masked <= refine_threshold
    low_a = A[low_mask]
    low_b = B[low_mask]
    da = (A_MAX - A_MIN) / GRID_SIZE
    db = (B_MAX - B_MIN) / GRID_SIZE
    a2_min, a2_max = low_a.min() - da, low_a.max() + da
    b2_min, b2_max = low_b.min() - db, low_b.max() + db

    # 在这个小范围内创建加密网格
    n_a2 = int((a2_max - a2_min) / da * REFINE_FACTOR) + 1
    n_b2 = int((b2_max - b2_min) / db * REFINE_FACTOR) + 1
    a2_values = np.linspace(a2_min, a2_max, n_a2)
    b2_values = np.linspace(b2_min, b2_max, n_b2)
    A2, B2 = np.meshgrid(a2_values, b2_values)
    Z2 = np.full_like(A2, np.nan)

    tasks2 = []
    task_indices2 = []
    reused_count = 0
    for i2, j2 in product(range(n_b2), range(n_a2)):
        a, b = A2[i2, j2], B2[i2, j2]
        i_orig = int((b - B_MIN) / db)
        j_orig = int((a - A_MIN) / da)
        i_orig = max(0, min(i_orig, GRID_SIZE - 1))
        j_orig = max(0, min(j_orig, GRID_SIZE - 1))
        if Z_masked[i_orig, j_orig] <= refine_threshold:
            if np.isclose(a, A[i_orig, j_orig]) and np.isclose(b, B[i_orig, j_orig]):
                Z2[i2, j2] = Z_masked[i_orig, j_orig]
                reused_count += 1
            else:
                tasks2.append(integrate_error.remote(a, b))
                task_indices2.append((i2, j2))

    print(
        f"加密区域 (threshold={refine_threshold:.4f}): 计算 {len(tasks2)} 个点, 复用 {reused_count} 个点"
    )
    results2 = ray.get(tasks2)
    for (i2, j2), result in zip(task_indices2, results2):
        Z2[i2, j2] = result
    Z2_masked = np.ma.masked_invalid(Z2)

    # 画双图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

    # 左图：全局
    contour1 = ax1.contourf(A, B, Z_masked, levels=20, cmap="viridis")
    contour_lines1 = ax1.contour(
        A,
        B,
        Z_masked,
        levels=np.linspace(0, 0.5, 21),
        colors="white",
        alpha=0.3,
        linewidths=0.5,
    )
    ax1.clabel(contour_lines1, inline=True, fontsize=8, fmt="%.3f")
    cbar1 = fig.colorbar(contour1, ax=ax1)
    cbar1.set_label(ERROR_LABEL, fontsize=12)
    ax1.set_title(f"Global: {ERROR_LABEL}", fontsize=14)
    ax1.set_xlabel("a", fontsize=12)
    ax1.set_ylabel("b", fontsize=12)
    ax1.grid(True, alpha=0.3, which="major")
    ax1.grid(True, alpha=0.15, which="minor")
    ax1.minorticks_on()
    ax1.plot(A_OPT, B_OPT, "rx", markersize=10, markeredgewidth=2)

    # 右图：加密区域
    contour2 = ax2.contourf(
        A2, B2, Z2_masked, levels=np.linspace(0, refine_threshold, 21), cmap="viridis"
    )
    contour_lines2 = ax2.contour(
        A2,
        B2,
        Z2_masked,
        levels=np.linspace(0, refine_threshold, 21),
        colors="white",
        alpha=0.3,
        linewidths=0.5,
    )
    ax2.clabel(contour_lines2, inline=True, fontsize=8, fmt="%.4f")
    cbar2 = fig.colorbar(contour2, ax=ax2)
    cbar2.set_label(ERROR_LABEL, fontsize=12)
    ax2.set_title(f"Refined (< {refine_threshold:.3f}): {ERROR_LABEL}", fontsize=14)
    ax2.set_xlabel("a", fontsize=12)
    ax2.set_ylabel("b", fontsize=12)
    ax2.grid(True, alpha=0.3, which="major")
    ax2.grid(True, alpha=0.15, which="minor")
    ax2.minorticks_on()
    ax2.plot(A_OPT, B_OPT, "rx", markersize=10, markeredgewidth=2)

    ab_opt_to_plot = ab_opt if ab_opt is not None else find_ab_opt(A2, B2, Z2_masked)
    fmt = ".4f" if ab_opt is None else ".4f"
    for a, b in ab_opt_to_plot:
        ax2.plot(a, b, "b^", markersize=10, markeredgewidth=2)
        ax2.text(
            a, b, f"  a={a:{fmt}}, b={b:{fmt}}", fontsize=9, ha="left", va="bottom"
        )

    fig.tight_layout()

    output_path = (
        Path(__file__).parent / "outputs" / f"parameter_contour_{FILE_SUFFIX}.png"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)

    print(f"\nSaved: {output_path}")
    print(f"Integral range: [{Z_masked.min():.6f}, {Z_masked.max():.6f}]")


if __name__ == "__main__":
    plot_parameter_contour()

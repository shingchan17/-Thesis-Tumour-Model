"""Generate placeholder .dat files so the LaTeX document compiles.
These are physically reasonable示意数据, not real simulation output.
Run: python make_placeholder_data.py
"""
import numpy as np
import os

os.makedirs("figs/data", exist_ok=True)

# ---- 1. D0c_profiles_RoverL.dat: 营养剖面 n/n∞ = (R/r) * sinh(r/L) / sinh(R/L)
# x = r/R, 三条曲线对应 R/L = 1, 5, 30
r = np.linspace(0.001, 1.0, 100)
def profile(R_over_L):
    L_ratio = R_over_L  # R/L
    # n/n∞ = (R/r) * sinh(r/L) / sinh(R/L)
    # 令 rho = r/R, 则 r/L = rho * R/L, R/r = 1/rho
    rho = r
    r_over_L = rho * L_ratio
    return (1.0 / rho) * np.sinh(r_over_L) / np.sinh(L_ratio)

with open("figs/data/D0c_profiles_RoverL.dat", "w") as f:
    for i in range(len(r)):
        f.write(f"{r[i]:.6f} {profile(1.0)[i]:.6f} {profile(5.0)[i]:.6f} {profile(30.0)[i]:.6f}\n")

# ---- 2. D2_growth_curve.dat: 生长曲线
# x=t, y1=R(t), y2=? (unused), y3=R_nec(t)
t = np.arange(0, 32, 0.5)
# 前沿: 近似 R ~ 2.0*t + 2.0 (初期慢, 后期趋近 2√δ=2)
R_front = 2.0 * t + 2.0 * (1 - np.exp(-t/5))
# 坏死核心: t<12 时为 0, 之后线性增长
R_nec = np.where(t < 12, 0.0, 2.0 * (t - 12))
with open("figs/data/D2_growth_curve.dat", "w") as f:
    for i in range(len(t)):
        f.write(f"{t[i]:.3f} {R_front[i]:.4f} 0.0 {R_nec[i]:.4f}\n")

# ---- 3. D3_profiles.dat: 空间剖面, 21 列 (每个时刻 3 列: r, c, n)
# 时刻: t=2,6,10,15,20,25,30
r = np.linspace(0, 55, 200)
times = [2, 6, 10, 15, 20, 25, 30]
R_front_t = {2: 3.7, 6: 9.97, 10: 16.97, 15: 26.04, 20: 35.28, 25: 44.62, 30: 54.03}
R_nec_t = {2: 0, 6: 0, 10: 0, 15: 4.8, 20: 15.55, 25: 25.05, 30: 34.55}

with open("figs/data/D3_profiles.dat", "w") as f:
    for j in range(len(r)):
        row = []
        for tk in times:
            Rf = R_front_t[tk]
            Rn = R_nec_t[tk]
            # c: 前沿用 tanh 过渡, 内部若 r<Rn 则坏死(c≈0), 否则 c≈1
            if r[j] < Rn:
                c = 0.05
            else:
                c = 0.5 * (1 - np.tanh((r[j] - Rf) / 1.5))
            # n: 内部低, 外部回到 1
            n = 1.0 / (1.0 + np.exp(-(r[j] - Rf + 3) / 2.0))
            # 每个时刻 3 列: r, c, n (共 7*3=21 列, 与 LaTeX 中 x index/y index 对应)
            row.append(f"{r[j]:.4f}")
            row.append(f"{c:.6f}")
            row.append(f"{n:.6f}")
        f.write(" ".join(row) + "\n")

# ---- 4. A2_rw_gauss.dat: 随机游走经验分布 (散点)
x = np.linspace(-30, 30, 61)
D = 0.5
t = 400
gauss = np.exp(-x**2 / (4 * D * t)) / np.sqrt(4 * np.pi * D * t)
with open("figs/data/A2_rw_gauss.dat", "w") as f:
    for i in range(len(x)):
        f.write(f"{x[i]:.2f} {gauss[i]:.6e}\n")

# ---- 5. C3_speed_curve.dat: 理论波速曲线 c* = 2√D
D_arr = np.linspace(0.1, 3.0, 100)
c_theory = 2 * np.sqrt(D_arr)
with open("figs/data/C3_speed_curve.dat", "w") as f:
    for i in range(len(D_arr)):
        f.write(f"{D_arr[i]:.4f} {c_theory[i]:.6f}\n")

# ---- 6. C2_wave_speed.dat: 数值波速点
D_pts = [0.5, 1.0, 2.0]
c_num = [1.347, 1.907, 2.699]
with open("figs/data/C2_wave_speed.dat", "w") as f:
    for d, c in zip(D_pts, c_num):
        f.write(f"{d:.4f} {c:.6f}\n")

# ---- 7. C4_collapse.dat: 行波塌缩
z = np.linspace(-10, 55, 200)
# U(z) = 1/(1+exp(z)) 形式的行波
U = 1.0 / (1.0 + np.exp(0.8 * z))
with open("figs/data/C4_collapse.dat", "w") as f:
    for i in range(len(z)):
        f.write(f"{z[i]:.4f} {U[i]:.6f} {U[i]:.6f} {U[i]:.6f}\n")

print("All placeholder .dat files generated in figs/data/")

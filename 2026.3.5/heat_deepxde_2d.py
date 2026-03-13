"""
DeepXDE 进阶教学：2D 热传导方程
=================================================
从 1D 方程扩展到 2D 空间域，学习 Rectangle 几何体与多维 Hessian 的使用。

目标方程（2D Heat Equation）：
  dT/dt = alpha * (d²T/dx² + d²T/dy²)      (x,y) ∈ [0,1]×[0,1],  t ∈ [0,1]

边界条件 (Dirichlet)：
  T = 0  在所有四条边上 (x=0, x=1, y=0, y=1)

初始条件：
  T(x, y, 0) = sin(pi*x) * sin(pi*y)

解析解：
  T(x, y, t) = sin(pi*x) * sin(pi*y) * exp(-2 * alpha * pi^2 * t)

输入 x 张量各列说明：
  x[:, 0:1] -> 空间坐标 x
  x[:, 1:2] -> 空间坐标 y
  x[:, 2:3] -> 时间坐标 t
"""

import deepxde as dde
import numpy as np
import matplotlib
matplotlib.use("Agg")          # 无 GUI 环境下直接保存，不弹窗
import matplotlib.pyplot as plt

# ==================== [1. 物理参数] ====================
alpha = 0.05

# ==================== [2. 时空几何域定义] ====================
# 2D 矩形空间域：x ∈ [0,1], y ∈ [0,1]
geom = dde.geometry.Rectangle([0, 0], [1, 1])

# 时间域：t ∈ [0,1]
timedomain = dde.geometry.TimeDomain(0, 1)

# 组合为 2D 时空域（输入张量形状：[N, 3]，三列分别为 x, y, t）
geomtime = dde.geometry.GeometryXTime(geom, timedomain)


# ==================== [3. PDE 残差函数] ====================
# 关键变化：
#   - dT/dt  : j=2（现在 t 是第 2 列，不是 1D 时的第 1 列）
#   - d²T/dx²: i=0, j=0（x 对 x 的 Hessian）
#   - d²T/dy²: i=1, j=1（y 对 y 的 Hessian）—— 2D 新增！
def pde(x, y):
    dT_dt    = dde.grad.jacobian(y, x, i=0, j=2)       # ∂T/∂t
    d2T_dx2  = dde.grad.hessian(y, x, i=0, j=0)        # ∂²T/∂x²
    d2T_dy2  = dde.grad.hessian(y, x, i=1, j=1)        # ∂²T/∂y²（2D 扩展项）
    return dT_dt - alpha * (d2T_dx2 + d2T_dy2)


# ==================== [4. 边界条件 & 初始条件] ====================
# DirichletBC：在矩形四条边上 T=0
#   Rectangle 的 on_boundary 自动匹配 x=0, x=1, y=0, y=1，无需手动区分
bc = dde.icbc.DirichletBC(
    geomtime,
    lambda x: 0,
    lambda _, on_boundary: on_boundary
)

# IC：T(x, y, 0) = sin(pi*x) * sin(pi*y)
#   注意：x[:, 0:1] 是 x 坐标，x[:, 1:2] 是 y 坐标
ic = dde.icbc.IC(
    geomtime,
    lambda x: np.sin(np.pi * x[:, 0:1]) * np.sin(np.pi * x[:, 1:2]),
    lambda _, on_initial: on_initial
)


# ==================== [5. 数据集与模型] ====================
data = dde.data.TimePDE(
    geomtime,
    pde,
    [bc, ic],
    num_domain=2000,    # 2D 域面积是 1D 的 2 倍，采样点相应增加
    num_boundary=200,   # 矩形有四条边，比 1D 两个端点多
    num_initial=200
)

# 网络输入维度：3（x, y, t），输出维度：1（温度 T）
net = dde.nn.FNN([3] + [64] * 4 + [1], "tanh", "Glorot normal")
model = dde.Model(data, net)


# ==================== [6. 两阶段训练] ====================
# Stage 1：Adam 粗调 + 动态重采样
pde_resampler = dde.callbacks.PDEPointResampler(period=100)

print("Stage 1: Adam 优化器粗调（5000 轮）...")
model.compile("adam", lr=1e-3)
losshistory, train_state = model.train(
    iterations=5000,
    callbacks=[pde_resampler]
)

# Stage 2：L-BFGS 精准微调
print("-" * 50)
print("Stage 2: L-BFGS 二阶优化精准微调...")
model.compile("L-BFGS")
losshistory, train_state = model.train(
    callbacks=[pde_resampler]
)


# ==================== [7. 精度验证：与解析解对比] ====================
print("-" * 50)
print("验证阶段：与解析解 T = sin(pi*x)*sin(pi*y)*exp(-2*alpha*pi^2*t) 对比...")

# 构建均匀验证网格（30x30 空间 × 固定时刻 t=0.5）
n = 30
x_lin = np.linspace(0, 1, n)
y_lin = np.linspace(0, 1, n)
XX, YY = np.meshgrid(x_lin, y_lin)

t_val = 0.5
TT = np.full_like(XX, t_val)

# 拼装为 [N, 3] 的输入矩阵
X_test = np.column_stack([XX.flatten(), YY.flatten(), TT.flatten()])

# 网络预测
T_pred = model.predict(X_test)

# 解析解
T_exact = (
    np.sin(np.pi * XX.flatten()) *
    np.sin(np.pi * YY.flatten()) *
    np.exp(-2 * alpha * np.pi**2 * t_val)
)

# 相对 L2 误差
rel_l2 = np.linalg.norm(T_pred.flatten() - T_exact) / np.linalg.norm(T_exact)
print(f"t={t_val} 时刻，空间网格 {n}×{n}，相对 L2 误差：{rel_l2:.4e}")


# ==================== [8. 可视化：三图并排保存为 PNG] ====================
T_pred_2d  = T_pred.flatten().reshape(n, n)
T_exact_2d = T_exact.reshape(n, n)
T_err_2d   = np.abs(T_pred_2d - T_exact_2d)   # 绝对误差

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle(
    f"2D Heat Equation (DeepXDE) — t = {t_val},  α = {alpha}\n"
    f"Relative L2 Error = {rel_l2:.4e}",
    fontsize=13
)

titles  = ["Exact Solution", "PINN Prediction", "Absolute Error"]
datas   = [T_exact_2d, T_pred_2d, T_err_2d]
cmaps   = ["hot",      "hot",     "coolwarm"]

for ax, title, data, cmap in zip(axes, titles, datas, cmaps):
    im = ax.imshow(
        data,
        origin="lower",
        extent=[0, 1, 0, 1],
        cmap=cmap,
        aspect="equal"
    )
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

plt.tight_layout()
save_path = "heat_2d_error.png"
plt.savefig(save_path, dpi=150, bbox_inches="tight")
print(f"图像已保存至：{save_path}")

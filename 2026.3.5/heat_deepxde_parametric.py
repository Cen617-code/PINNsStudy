"""
DeepXDE 高阶教学：参数化 PINNs 代理模型 (Parametric PINN)
======================================================
目标：将热扩散系数 α 作为一个输入参数，训练一个能适应不同 α 值的代理模型。
对标 Phase 3 的 `multi_alpha_validation.py`。

核心维度变化：
- 手写 PyTorch：`torch.cat([x, t, alpha], dim=1)` -> 网络输入 3 维
- DeepXDE：将参数空间直接耦合进几何域，变成 3D 几何，输入依然是由 DeepXDE 管理的 `x`，但 `x.shape[1] == 3`
"""

import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt

# 参数区间
alpha_min = 0.01
alpha_max = 0.1

# ==================== [引导问题 5：参数化时空域的构建] ====================
# 在普通 1D 热传导中，几何域是 Interval(0, 1) [空间] × TimeDomain(0, 1) [时间]。共 2 维。
# 现在我们要把 α 也当作一个“坐标轴”。这样，每次在域内随机采点时，都会自动采样到一个 α。
# 
# 所以，空间部分变成了 2 维：x 轴是位置，y 轴是 α 参数！
# 提示：DeepXDE 用 `dde.geometry.Rectangle(xmin, xmax)` 表示 2D 矩形域。
# 其中 `xmin` 和 `xmax` 都是长度为 2 的列表：[位置下界, α下界] 和 [位置上界, α上界]。

# 请在此补全 Rectangle 定义：
# geom_2d = dde.geometry.Rectangle([0, ???], [1, ???])
geom_2d = dde.geometry.Rectangle([0, alpha_min], [1, alpha_max])

timedomain = dde.geometry.TimeDomain(0, 1)

# 将 "空间(含α) + 时间" 组合成 3D 时空域 
geomtime = dde.geometry.GeometryXTime(geom_2d, timedomain)

# ==================== [引导问题 6：参数化 PDE 的残差] ====================
# 网络现在的输入 x 有 3 列：
# x[:, 0] -> 空间位置 x
# x[:, 1] -> 热扩散系数 α (这是因为我们在 Rectangle 里把它放在了第二维)
# x[:, 2] -> 时间 t (因为 GeometryXTime 总是把时间放在最后一列)
def pde(x, y):
    # 提取当前数据点的 α 矩阵
    alpha_val = x[:, 1:2] 
    
    # 提取导数 (注意找对时间 t 和 空间 x 对应的列索引 j！)
    dT_dt = dde.grad.jacobian(y, x, i=0, j=2)
    d2T_dx2 = dde.grad.hessian(y, x, i=0, j=0)
    
    return dT_dt - alpha_val * d2T_dx2
    pass 

# ==================== [边界与初始条件] ====================
# T(0, α, t) = 0 和 T(1, α, t) = 0
# DeepXDE 的 on_boundary 会自动处理矩形的 4 条边，但我们只想对 x=0 和 x=1 施加 BC
# 所以这里加一个判断函数：(我们提供现成的)
def boundary_x(x, on_boundary):
    return on_boundary and (np.isclose(x[0], 0) or np.isclose(x[0], 1))

bc = dde.icbc.DirichletBC(geomtime, lambda x: 0, boundary_x)
ic = dde.icbc.IC(geomtime, lambda x: np.sin(np.pi * x[:, 0:1]), lambda _, on_initial: on_initial)

# ==================== [模型组装与训练] ====================
# 参数化问题需要更多采样点和适当宽一点的网络
data = dde.data.TimePDE(
    geomtime, pde, [bc, ic], 
    num_domain=5000, num_boundary=500, num_initial=500
)

# 网络输入变为了 [3]！
net = dde.nn.FNN([3] + [64] * 3 + [1], "tanh", "Glorot normal")

model = dde.Model(data, net)
pde_resampler = dde.callbacks.PDEPointResampler(period=100)

model.compile("adam", lr=1e-3)
print("开始参数化模型 Adam 粗调...")
model.train(iterations=10000, callbacks=[pde_resampler])

model.compile("L-BFGS")
print("开始 L-BFGS 微调...")
model.train(callbacks=[pde_resampler])


# ==================== [泛化验证与可视化] ====================
print("-" * 50)
print("开始多 α 泛化验证...")

# 选取 8 个不同 α 进行验证
alpha_values = np.linspace(alpha_min, alpha_max, 8)
errors = []

# 生成 100x100 的时空网格 (x, t)
x_eval = np.linspace(0, 1, 100)
t_eval = np.linspace(0, 1, 100)
X, T_grid = np.meshgrid(x_eval, t_eval, indexing='ij')

X_flat = X.flatten()[:, None]
T_flat = T_grid.flatten()[:, None]

for alpha_val in alpha_values:
    # 构建当前 α 的输入矩阵，尺寸: [10000, 3]，三列分别为 x, α, t
    alpha_flat = np.full_like(X_flat, alpha_val)
    XTA_eval = np.hstack((X_flat, alpha_flat, T_flat))
    
    # DeepXDE 的预测接口非常简单，直接传入 NumPy 数组
    T_pred_flat = model.predict(XTA_eval)
    T_pred = T_pred_flat.reshape(100, 100)
    
    # 解析解公式: T(x,t) = sin(πx) · exp(-α·π²·t)
    T_exact = np.sin(np.pi * X) * np.exp(-alpha_val * (np.pi**2) * T_grid)
    
    # 计算相对 L2 误差
    error_l2 = np.linalg.norm(T_pred - T_exact, 2) / np.linalg.norm(T_exact, 2)
    errors.append(error_l2)
    print(f"α = {alpha_val:.4f} | 相对 L2 误差 = {error_l2:.4e}")

# 画图
plt.figure(figsize=(8, 5))
plt.plot(alpha_values, errors, marker='o', linestyle='-', color='#d62728', linewidth=2, markersize=8, label='DeepXDE (L-BFGS)')
plt.xlabel(r'Thermal Diffusivity ($\alpha$)', fontsize=12)
plt.ylabel('Relative L2 Error', fontsize=12)
plt.title('DeepXDE Parametric PINN Generalization Error', fontsize=14)
plt.yscale('log')
plt.grid(True, which="both", ls="--", alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig('deepxde_multi_alpha_error.png', dpi=150, bbox_inches='tight')
print("多 α 验证图已保存为 'deepxde_multi_alpha_error.png'")


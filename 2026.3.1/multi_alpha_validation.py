"""
参数化 PINN (Parametric Physics-Informed Neural Network)
========================================================
目标: 求解参数化 1D 热传导方程 dT/dt = α · d²T/dx²
      其中 α (热扩散系数) 作为网络输入维度, 使同一个网络能预测不同 α 下的温度场。
      这是从"求解器"到"代理模型 (Surrogate Model)"的核心跃迁。

方程:  dT/dt = α · d²T/dx²,  x ∈ [0,1], t ∈ [0,1]
初始条件 (IC): T(x, 0) = sin(πx)
边界条件 (BC): T(0, t) = 0, T(1, t) = 0
解析解: T(x,t) = sin(πx) · exp(-α·π²·t)

Phase 3 学习代码 — 2026.2.28
"""

import torch
import torch.nn as nn
import torch.optim as optim
import math
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

# ===================== 超参数配置 =====================
N_f = 1000              # PDE 域内配置点数量
N_ic = 100              # 初始条件采样点数量
N_bc = 100              # 边界条件采样点数量（每侧）
alpha_min = 0.01        # α 采样下界
alpha_max = 0.1         # α 采样上界
alpha_test = 0.05       # 验证时使用的固定 α 值
epochs = 10000          # 训练轮数
learning_rate = 1e-3    # Adam 学习率
w_data = 1

# ===================== 采样配置点 =====================
# 内部点: 需要 requires_grad=True (x, t), 因为 PDE 残差需要对它们求导
# α 不需要 requires_grad, 它只是系数, 不是被微分的自变量
x = torch.rand(N_f, 1, requires_grad=True)
t = torch.rand(N_f, 1, requires_grad=True)
alpha_sample = alpha_min + (alpha_max - alpha_min) * torch.rand(N_f, 1)

# 初始条件: t=0, x 随机, α 随机
x_ic = torch.rand(N_ic, 1)
t_ic = torch.zeros(N_ic, 1)
alpha_ic = alpha_min + (alpha_max - alpha_min) * torch.rand(N_ic, 1)

# 左边界: x=0, t 随机, α 随机
x_b = torch.zeros(N_bc, 1)
t_b = torch.rand(N_bc, 1)
alpha_b = alpha_min + (alpha_max - alpha_min) * torch.rand(N_bc, 1)

# 右边界: x=1, t 随机, α 随机
x_c = torch.ones(N_bc, 1)
t_c = torch.rand(N_bc, 1)
alpha_c = alpha_min + (alpha_max - alpha_min) * torch.rand(N_bc, 1)

# ===================== 观测数据采样 =====================
N_data = 200        #域内观测点数量（远少于配置点 N_f=1000）

x_data = torch.rand(N_data, 1)
t_data = torch.rand(N_data, 1)
alpha_data = alpha_min + (alpha_max - alpha_min) * torch.rand(N_data, 1)

# "观测值" — 由解析解生成（模拟 COMSOL 或实验数据）
T_data_target =   torch.sin(math.pi * x_data) * torch.exp(-alpha_data * (math.pi**2) * t_data)

# ===================== 网络定义 =====================
class SimpleMLP(nn.Module):
    """可配置的多层感知机, 作为温度场 T(x, t, α) 的通用逼近器。
    使用 nn.Sequential 动态构建任意层数的隐藏层。
    """
    def __init__(self, input_dim=3, hidden_dim=64, output_dim=1, num_hidden_layers=4):
        super().__init__()

        # 动态构建隐藏层: [input_dim → hidden_dim → ... → hidden_dim]
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.Tanh())  # PINNs 必须用平滑激活函数, 保证高阶导数不为零

        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())

        self.hidden_layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # 数据直接穿过所有隐藏层(Sequential 自动按顺序执行)
        x = self.hidden_layers(x)
        # 最后通过输出层 (无激活函数, 输出温度值不受限)
        output = self.output_layer(x)
        return output


# ===================== PDE 残差计算 =====================
def compute_pde_residual(model, x, t, alpha):
    """计算热传导方程残差: residual = dT/dt - α · d²T/dx²
    
    Args:
        model: 神经网络模型
        x: 空间坐标, shape (N, 1), requires_grad=True
        t: 时间坐标, shape (N, 1), requires_grad=True
        alpha: 热扩散系数张量, shape (N, 1), 逐元素参与乘法
    """
    # 网络输入是 3 维: (x, t, α)
    T = model.forward(torch.cat([x, t, alpha], dim=1))

    # 一阶时间导数 dT/dt
    dT_dt = torch.autograd.grad(
        outputs=T, inputs=t,
        grad_outputs=torch.ones_like(T),
        create_graph=True)[0]

    # 一阶空间导数 dT/dx
    dT_dx = torch.autograd.grad(
        outputs=T, inputs=x,
        grad_outputs=torch.ones_like(T),
        create_graph=True)[0]

    # 二阶空间导数 d²T/dx² (拉普拉斯算子项)
    d2T_dx2 = torch.autograd.grad(
        outputs=dT_dx, inputs=x,
        grad_outputs=torch.ones_like(T),
        create_graph=True)[0]

    # 残差应趋近于 0: 热传导方程被满足时, dT/dt == α · d²T/dx²
    # 注意: alpha 是 (N,1) 张量, 与 d2T_dx2 做逐元素乘法
    residual = dT_dt - alpha * d2T_dx2
    return residual


# ===================== 训练 =====================
model = SimpleMLP()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.5)

print("开始训练...")
for epoch in range(epochs):
    optimizer.zero_grad()

    # PDE 损失: 域内物理方程残差 → 0
    residual = compute_pde_residual(model, x, t, alpha_sample)
    loss_pde = F.mse_loss(residual, torch.zeros_like(residual))

    # IC 损失: T(x, 0, α) = sin(πx), 初始条件与 α 无关
    loss_ic = F.mse_loss(
        model(torch.cat([x_ic, t_ic, alpha_ic], dim=1)),
        torch.sin(x_ic * math.pi))

    # 数据损失：网络预测 vs "观测值"
    loss_data = F.mse_loss(
        model(torch.cat([x_data, t_data, alpha_data,], dim=1)),
        T_data_target)
    
    # BC 损失: T(0, t, α) = 0, T(1, t, α) = 0
    T_left = model(torch.cat([x_b, t_b, alpha_b], dim=1))
    T_right = model(torch.cat([x_c, t_c, alpha_c], dim=1))
    loss_bc = (F.mse_loss(T_left, torch.zeros_like(T_left))
               + F.mse_loss(T_right, torch.zeros_like(T_right)))

    # 等权联合训练
    loss_total = loss_pde + loss_ic + loss_bc + w_data * loss_data

    loss_total.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss Total: {loss_total.item():.6f} "
              f"(PDE: {loss_pde.item():.6f}, IC: {loss_ic.item():.6f}, BC: {loss_bc.item():.6f}), Data: {loss_data.item():.6f}")
print("训练完成！")


# ===================== 验证与可视化 =====================
alpha_values = np.linspace(alpha_min, alpha_max, 8)
errors = []

x_eval = torch.linspace(0, 1, 100)
t_eval = torch.linspace(0, 1, 100)
X, T_grid = torch.meshgrid(x_eval, t_eval, indexing='ij')

X_flat = X.flatten().unsqueeze(1)
T_flat = T_grid.flatten().unsqueeze(1)

model.eval()
for alpha_val in alpha_values:
    alpha_flat = torch.full_like(X_flat, alpha_val)

    XTA_eval = torch.cat([X_flat, T_flat, alpha_flat], dim=1)

    with torch.no_grad():
        T_pred_flat = model(XTA_eval)

    T_pred = T_pred_flat.reshape(100, 100).numpy()
    X_np = X.numpy()
    T_np = T_grid.numpy()

    # 解析解: T(x,t) = sin(πx) · exp(-α·π²·t)
    T_exact = np.sin(np.pi * X_np) * np.exp(-alpha_val * (np.pi**2) * T_np)
    error_l2 = np.linalg.norm(T_pred - T_exact, 2) / np.linalg.norm(T_exact, 2)
    errors.append(error_l2)

    print(f"α = {alpha_val:.4f} | 相对 L2 误差 = {error_l2:.4e}")

plt.figure(figsize=(8,5))
plt.plot(alpha_values, errors, marker='o', linestyle='-', color='#1f77b4', linewidth=2, markersize=8)
plt.xlabel(r'Thermal Diffusivity ($\alpha$)', fontsize=12)
plt.ylabel('Relative L2 Error', fontsize=12)
plt.title('PINN Generalization over Multiple $\\alpha$ Values', fontsize=14)
plt.yscale('log')
plt.grid(True, which="both", ls="--", alpha=0.6)
plt.tight_layout()
plt.savefig('multi_alpha_error.png', dpi=150, bbox_inches='tight')
print("多 α 验证图已保存为 'multi_alpha_error.png'")
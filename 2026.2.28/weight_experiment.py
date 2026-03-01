"""
Loss 权重平衡实验 (Weight Balancing Experiment)
================================================
目标: 在 1D 热传导方程 PINN 上, 实验不同 Loss 权重策略对收敛精度的影响。

实验内容:
  1. 手动固定权重对比 (修改 w_pde, w_ic, w_bc 后重跑)
  2. 自适应权重 (基于梯度范数自动平衡各项 Loss 贡献)

关键结论:
  - 简单问题上三项 Loss 天然量级接近, 等权 (1:1:1) 即为近似最优
  - 人为破坏平衡会让某些约束被系统性忽略
  - 自适应方法的价值在"各项 Loss 天然量级差异巨大"的复杂多物理场问题中

Phase 2 学习代码 — 2026.2.28
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
alpha = 0.01            # 热扩散系数 (固定)
epochs = 5000           # 训练轮数
learning_rate = 1e-3    # Adam 学习率
# 手动权重 — 修改这三个值来进行对比实验
w_pde = 1
w_ic = 1
w_bc = 1

# ===================== 采样配置点 =====================
# 内部点: x, t ∈ [0,1] 均匀随机, 需要 requires_grad 以支持 autograd 求 PDE 残差
x = torch.rand(N_f, 1, requires_grad=True)
t = torch.rand(N_f, 1, requires_grad=True)

# IC 点: t=0, x 随机 (不需要 requires_grad, 只做数据拟合)
x_ic = torch.rand(N_ic, 1)
t_ic = torch.zeros(N_ic, 1)

# BC 左边界: x=0
x_b = torch.zeros(N_bc, 1)
t_b = torch.rand(N_bc, 1)

# BC 右边界: x=1
x_c = torch.ones(N_bc, 1)
t_c = torch.rand(N_bc, 1)


# ===================== 网络定义 =====================
class SimpleMLP(nn.Module):
    """单隐层 MLP, 作为温度场 T(x,t) 的通用逼近器。"""
    def __init__(self):
        super().__init__()
        self.hidden_layer = nn.Linear(2, 50)    # 输入 (x,t) → 50 个隐藏神经元
        self.activation = nn.Tanh()              # 平滑激活函数 (PINNs 禁用 ReLU)
        self.output_layer = nn.Linear(50, 1)     # 50 → 输出温度 T

    def forward(self, x):
        x = self.hidden_layer(x)
        x = self.activation(x)
        output = self.output_layer(x)
        return output


# ===================== PDE 残差计算 =====================
def compute_pde_residual(model, x, t, alpha):
    """计算热传导方程残差: residual = dT/dt - α · d²T/dx²"""
    T = model.forward(torch.cat([x, t], dim=1))

    dT_dt = torch.autograd.grad(
        outputs=T, inputs=t,
        grad_outputs=torch.ones_like(T),
        create_graph=True)[0]

    dT_dx = torch.autograd.grad(
        outputs=T, inputs=x,
        grad_outputs=torch.ones_like(T),
        create_graph=True)[0]

    d2T_dx2 = torch.autograd.grad(
        outputs=dT_dx, inputs=x,
        grad_outputs=torch.ones_like(T),
        create_graph=True)[0]

    residual = dT_dt - alpha * d2T_dx2
    return residual


# ===================== 自适应权重计算 =====================
def compute_adaptive_weights(model, loss_pde, loss_ic, loss_bc):
    """基于梯度范数自动平衡各项 Loss 权重。
    
    原理: 梯度范数大的项 → 已在主导训练 → 降低权重
          梯度范数小的项 → 被压制 → 提高权重
    
    注意:
      - 使用 retain_graph=True (保留计算图, 供后续 backward 使用)
      - 不需要 create_graph=True (梯度范数仅是统计量, 不参与反向传播)
      - 结果必须 .detach() 脱离计算图
    """
    # 分别计算各项 Loss 对网络最后一层权重的梯度
    grad_pde_tensor = torch.autograd.grad(loss_pde, model.output_layer.weight, retain_graph=True)[0]
    grad_ic_tensor  = torch.autograd.grad(loss_ic,  model.output_layer.weight, retain_graph=True)[0]
    grad_bc_tensor  = torch.autograd.grad(loss_bc,  model.output_layer.weight, retain_graph=True)[0]

    # 计算 L2 范数
    grad_pde = torch.linalg.norm(grad_pde_tensor)
    grad_ic  = torch.linalg.norm(grad_ic_tensor)
    grad_bc  = torch.linalg.norm(grad_bc_tensor)

    # 反比例分配: w_i = mean / grad_i
    grad_mean = (grad_pde + grad_ic + grad_bc) / 3.0

    w_pde = (grad_mean / (grad_pde + 1e-8)).detach()   # 1e-8 防止除零
    w_ic  = (grad_mean / (grad_ic  + 1e-8)).detach()
    w_bc  = (grad_mean / (grad_bc  + 1e-8)).detach()

    return w_pde, w_ic, w_bc


# ===================== 训练 =====================
model = SimpleMLP()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

print("开始训练...")
for epoch in range(epochs):
    optimizer.zero_grad()

    # PDE 损失
    residual = compute_pde_residual(model, x, t, alpha)
    loss_pde = F.mse_loss(residual, torch.zeros_like(residual))

    # IC 损失: T(x, 0) = sin(πx)
    loss_ic = F.mse_loss(model(torch.cat([x_ic, t_ic], dim=1)), torch.sin(x_ic * math.pi))

    # BC 损失: T(0, t) = 0, T(1, t) = 0
    T_left = model(torch.cat([x_b, t_b], dim=1))
    T_right = model(torch.cat([x_c, t_c], dim=1))
    loss_bc = F.mse_loss(T_left, torch.zeros_like(T_left)) + F.mse_loss(T_right, torch.zeros_like(T_right))

    # 自适应权重 (注释掉下面一行可改回手动固定权重)
    w_pde, w_ic, w_bc = compute_adaptive_weights(model, loss_pde, loss_ic, loss_bc)
    loss_total = w_pde * loss_pde + w_ic * loss_ic + w_bc * loss_bc

    loss_total.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss Total: {loss_total.item():.6f} "
              f"(PDE: {loss_pde.item():.6f}, IC: {loss_ic.item():.6f}, BC: {loss_bc.item():.6f})")
print("训练完成！")


# ===================== 验证 =====================
x_eval = torch.linspace(0, 1, 100)
t_eval = torch.linspace(0, 1, 100)
X, T_grid = torch.meshgrid(x_eval, t_eval, indexing='ij')

X_flat = X.flatten().unsqueeze(1)
T_flat = T_grid.flatten().unsqueeze(1)
XT_eval = torch.cat([X_flat, T_flat], dim=1)

model.eval()
with torch.no_grad():
    T_pred_flat = model(XT_eval)

T_pred = T_pred_flat.reshape(100, 100).numpy()

X_np = X.numpy()
T_np = T_grid.numpy()
alpha_val = 0.01

# 解析解
T_exact = np.sin(np.pi * X_np) * np.exp(-alpha_val * (np.pi**2) * T_np)
error_l2 = np.linalg.norm(T_pred - T_exact, 2) / np.linalg.norm(T_exact, 2)
print(f"相对 L2 误差 (Relative L2 Error): {error_l2:.4e}")
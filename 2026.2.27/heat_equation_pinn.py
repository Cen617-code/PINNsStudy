"""
1D 热传导方程 PINN (1D Heat Equation Physics-Informed Neural Network)
======================================================================
目标: 求解 1D 热传导方程 dT/dt = α · d²T/dx²
      这是 Phase 2 的基础里程碑, 验证 PINNs 能够仅依靠物理残差和边界条件收敛到解析解。

方程:  dT/dt = α · d²T/dx²,  x ∈ [0,1], t ∈ [0,1]
初始条件 (IC): T(x, 0) = sin(πx)
边界条件 (BC): T(0, t) = 0, T(1, t) = 0
热扩散系数: α = 0.01 (常数)
解析解: T(x,t) = sin(πx) · exp(-α·π²·t)

Phase 2 学习代码 — 2026.2.27
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
N_bc = 100              # 边界条件采样点数量
alpha = 0.01            # 热扩散系数
epochs = 5000           # 训练轮数
learning_rate = 1e-3    # Adam 学习率

# ===================== 采样配置点 =====================
# 内部点: 定义域 (x,t) ∈ [0,1]×[0,1]
# 注意: 需要计算 autograd 的输入必须 requires_grad=True
x = torch.rand(N_f, 1, requires_grad=True)
t = torch.rand(N_f, 1, requires_grad=True)

# 初始条件点: (x, 0), x ∈ [0,1] 随机
x_ic = torch.rand(N_ic, 1)
t_ic = torch.zeros(N_ic, 1)

# 左边界条件点: (0, t), t ∈ [0,1] 随机
x_b = torch.zeros(N_bc, 1)
t_b = torch.rand(N_bc, 1)

# 右边界条件点: (1, t), t ∈ [0,1] 随机
x_c = torch.ones(N_bc, 1)
t_c = torch.rand(N_bc, 1)


# ===================== 网络模型 =====================
class SimpleMLP(nn.Module):
    """物理场逼近网络, 输入时空坐标 (x,t), 输出温度 T"""
    def __init__(self):
        super().__init__()
        # 定义输入层到隐藏层（输入维度2 -> 隐藏神经元50）
        self.hidden_layer = nn.Linear(2, 50)
        # 定义激活函数: 必须使用具备二阶平滑连续可导性质的函数 (不能用 ReLU!)
        self.activation = nn.Tanh()
        # 定义隐藏层到输出层（隐藏神经元50 -> 输出维度1）
        self.output_layer = nn.Linear(50, 1)

    def forward(self, x):
        # 数据流水线：输入 -> 隐藏层 -> 激活函数 -> 输出层
        x = self.hidden_layer(x)
        x = self.activation(x)
        output = self.output_layer(x)
        return output


# ===================== PDE物理方程残差 =====================
def compute_pde_residual(model, x, t, alpha):
    """使用 autograd 计算各阶导数, 组合成 PDE 残差"""
    # 获取网络预测温度 T
    T = model.forward(torch.cat([x, t], dim=1))
    
    # 计算 dT/dt (一阶时间导数)
    # create_graph=True: 保持计算图连通性, 以便反向传播计算梯度
    dT_dt = torch.autograd.grad(
        outputs=T, inputs=t, 
        grad_outputs=torch.ones_like(T), 
        create_graph=True)[0]
        
    # 计算 dT/dx (一阶空间导数)
    dT_dx = torch.autograd.grad(
        outputs=T, inputs=x, 
        grad_outputs=torch.ones_like(T),
        create_graph=True)[0]
        
    # 计算 d²T/dx² (二阶空间导数，对 dT_dx 再求导)
    d2T_dx2 = torch.autograd.grad(
        outputs=dT_dx, inputs=x,
        grad_outputs=torch.ones_like(T),
        create_graph=True)[0]
        
    # 残差项：如果物理定律被满足，该值应为 0
    residual = dT_dt - alpha * d2T_dx2
    return residual


# ===================== 训练过程 =====================
model = SimpleMLP()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

print("开始训练...")
for epoch in range(epochs):
    optimizer.zero_grad()

    # 1. 物理损失 L_pde: 让残差趋近于 0
    residual = compute_pde_residual(model, x, t, alpha)
    loss_pde = F.mse_loss(residual, torch.zeros_like(residual))
    
    # 2. 初始条件损失 L_ic: T(x,0) = sin(πx)
    loss_ic = F.mse_loss(model(torch.cat([x_ic, t_ic], dim=1)), torch.sin(x_ic * math.pi))
    
    # 3. 边界条件损失 L_bc: T(0,t) = 0, T(1,t) = 0
    T_left = model(torch.cat([x_b, t_b], dim=1))
    T_right = model(torch.cat([x_c, t_c], dim=1))
    loss_bc = F.mse_loss(T_left, torch.zeros_like(T_left)) + F.mse_loss(T_right, torch.zeros_like(T_right))

    # 总损失 = 等权相加 (因为在这个简单问题中三者量级相近)
    loss_total = loss_pde + loss_ic + loss_bc

    # 反向传播更新权重
    loss_total.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss Total: {loss_total.item():.6f} "
              f"(PDE: {loss_pde.item():.6f}, IC: {loss_ic.item():.6f}, BC: {loss_bc.item():.6f})")
print("训练完成！")


# ===================== 验证与可视化 =====================
# 生成均匀网格用于评估
x_eval = torch.linspace(0, 1, 100)
t_eval = torch.linspace(0, 1, 100)
X, T_grid = torch.meshgrid(x_eval, t_eval, indexing='ij')

X_flat = X.flatten().unsqueeze(1)
T_flat = T_grid.flatten().unsqueeze(1)
XT_eval = torch.cat([X_flat, T_flat], dim=1)

# 开启评估模式，关闭梯度计算
model.eval()
with torch.no_grad():
    T_pred_flat = model(XT_eval)

T_pred = T_pred_flat.reshape(100, 100).numpy()

# 计算解析解精确值
X_np = X.numpy()
T_np = T_grid.numpy()
alpha_val = 0.01
T_exact = np.sin(np.pi * X_np) * np.exp(-alpha_val * (np.pi**2) * T_np)

# 计算相对 L2 误差并打印
error_l2 = np.linalg.norm(T_pred - T_exact, 2) / np.linalg.norm(T_exact, 2)
print(f"相对 L2 误差 (Relative L2 Error): {error_l2:.4e}")

# ---- 绘图部分 ----
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
vmin = min(T_exact.min(), T_pred.min())
vmax = max(T_exact.max(), T_pred.max())

h1 = axes[0].imshow(T_exact.T, interpolation='nearest', cmap='rainbow', 
                    extent=[0, 1, 0, 1], origin='lower', aspect='auto', vmin=vmin, vmax=vmax)
axes[0].set_title('Exact Solution $T_{exact}(x,t)$')
axes[0].set_xlabel('x')
axes[0].set_ylabel('t')
fig.colorbar(h1, ax=axes[0])

h2 = axes[1].imshow(T_pred.T, interpolation='nearest', cmap='rainbow', 
                    extent=[0, 1, 0, 1], origin='lower', aspect='auto', vmin=vmin, vmax=vmax)
axes[1].set_title('PINN Prediction $T_{pred}(x,t)$')
axes[1].set_xlabel('x')
axes[1].set_ylabel('t')
fig.colorbar(h2, ax=axes[1])

abs_error = np.abs(T_exact - T_pred)
h3 = axes[2].imshow(abs_error.T, interpolation='nearest', cmap='viridis', 
                    extent=[0, 1, 0, 1], origin='lower', aspect='auto')
axes[2].set_title(f'Absolute Error (Rel L2: {error_l2:.2e})')
axes[2].set_xlabel('x')
axes[2].set_ylabel('t')
fig.colorbar(h3, ax=axes[2])

plt.tight_layout()
plt.savefig('heat_equation_result.png', dpi=150, bbox_inches='tight')
print("图片已保存为 heat_equation_result.png")
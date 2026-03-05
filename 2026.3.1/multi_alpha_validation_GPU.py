"""
参数化 PINN (Parametric Physics-Informed Neural Network) — H100 GPU 加速版
====================================================================
目标: 求解参数化 1D 热传导方程 dT/dt = α · d²T/dx²
"""

import torch
import torch.nn as nn
import torch.optim as optim
import math
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

# ===================== 新增：GPU 设备检测 =====================
# 自动检测是否存在 CUDA (GPU)，否则回退到 CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"==================================================")
print(f"🚀 当前使用的计算设备: {device}")
if device.type == 'cuda':
    print(f"🚀 显卡型号: {torch.cuda.get_device_name(0)}")
print(f"==================================================")

# ===================== 超参数配置 =====================
N_f = 1000              
N_ic = 100              
N_bc = 100              
alpha_min = 0.01        
alpha_max = 0.1         
alpha_test = 0.05       
epochs = 10000          
learning_rate = 1e-3    
w_data = 1              

# ===================== 观测数据采样 (直接生成在 GPU 上) =====================
N_data = 200        

# 新增：加上 device=device，直接在显存里生成数据
x_data = torch.rand(N_data, 1, device=device)
t_data = torch.rand(N_data, 1, device=device)
alpha_data = alpha_min + (alpha_max - alpha_min) * torch.rand(N_data, 1, device=device)

T_data_target = torch.sin(math.pi * x_data) * torch.exp(-alpha_data * (math.pi**2) * t_data)

# ===================== 网络定义 =====================
class SimpleMLP(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=64, output_dim=1, num_hidden_layers=4):
        super().__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.Tanh()) 

        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())

        self.hidden_layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.hidden_layers(x)
        output = self.output_layer(x)
        return output

# ===================== PDE 残差计算 =====================
def compute_pde_residual(model, x, t, alpha):
    T = model.forward(torch.cat([x, t, alpha], dim=1))

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

# ===================== 训练准备 =====================
# 新增：将模型整体搬运到 GPU 显存中 (.to(device))
model = SimpleMLP().to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.5)

print("开始训练...")
for epoch in range(epochs):
    # ===================== 动态重采样配置点 (全放到 GPU) =====================
    x = torch.rand(N_f, 1, requires_grad=True, device=device)
    t = torch.rand(N_f, 1, requires_grad=True, device=device)
    alpha_sample = alpha_min + (alpha_max - alpha_min) * torch.rand(N_f, 1, device=device)

    x_ic = torch.rand(N_ic, 1, device=device)
    t_ic = torch.zeros(N_ic, 1, device=device)
    alpha_ic = alpha_min + (alpha_max - alpha_min) * torch.rand(N_ic, 1, device=device)

    x_b = torch.zeros(N_bc, 1, device=device)
    t_b = torch.rand(N_bc, 1, device=device)
    alpha_b = alpha_min + (alpha_max - alpha_min) * torch.rand(N_bc, 1, device=device)

    x_c = torch.ones(N_bc, 1, device=device)
    t_c = torch.rand(N_bc, 1, device=device)
    alpha_c = alpha_min + (alpha_max - alpha_min) * torch.rand(N_bc, 1, device=device)

    optimizer.zero_grad()

    # PDE 损失
    residual = compute_pde_residual(model, x, t, alpha_sample)
    loss_pde = F.mse_loss(residual, torch.zeros_like(residual))

    # IC 损失
    loss_ic = F.mse_loss(
        model(torch.cat([x_ic, t_ic, alpha_ic], dim=1)),
        torch.sin(x_ic * math.pi))

    # 数据损失
    loss_data = F.mse_loss(
        model(torch.cat([x_data, t_data, alpha_data], dim=1)),
        T_data_target)
    
    # BC 损失
    T_left = model(torch.cat([x_b, t_b, alpha_b], dim=1))
    T_right = model(torch.cat([x_c, t_c, alpha_c], dim=1))
    loss_bc = (F.mse_loss(T_left, torch.zeros_like(T_left))
               + F.mse_loss(T_right, torch.zeros_like(T_right)))

    # 等权联合训练
    loss_total = loss_pde + loss_ic + loss_bc + w_data * loss_data

    loss_total.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss Total: {loss_total.item():.6e} "
              f"(PDE: {loss_pde.item():.6e}, IC: {loss_ic.item():.6e}, BC: {loss_bc.item():.6e}), Data: {loss_data.item():.6e}")

print("训练完成！开始画图...")

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

    # 验证数据丢给 GPU，并且不用计算梯度
    XTA_eval = torch.cat([X_flat, T_flat, alpha_flat], dim=1).to(device)

    with torch.no_grad():
        T_pred_flat = model(XTA_eval)

    # 新增：必须把计算结果从 GPU 内存 (.cpu()) 拉回普通内存，才能转成 NumPy
    T_pred = T_pred_flat.cpu().reshape(100, 100).numpy()
    X_np = X.numpy()
    T_np = T_grid.numpy()

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
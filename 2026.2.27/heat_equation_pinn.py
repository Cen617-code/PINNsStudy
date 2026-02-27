import torch 
import torch.nn as nn
import torch.optim as optim
import math
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

N_f = 1000
N_ic = 100
N_bc = 100
alpha = 0.01
epochs = 5000
learning_rate = 1e-3

x = torch.rand(N_f, 1, requires_grad=True)
t = torch.rand(N_f, 1, requires_grad=True)

x_ic = torch.rand(N_ic, 1)
t_ic = torch.zeros(N_ic, 1)

x_b = torch.zeros(N_bc, 1)
t_b = torch.rand(N_bc, 1)

x_c = torch.ones(N_bc, 1)
t_c = torch.rand(N_bc, 1)

class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        # 定义输入层到隐藏层（输入维度2 -> 隐藏神经元20）
        self.hidden_layer = nn.Linear(2, 50)
        # 定义激活函数
        self.activation = nn.Tanh()
        # 定义隐藏层到输出层（隐藏神经元20 -> 输出维度1）
        self.output_layer = nn.Linear(50, 1)

    def forward(self, x):
        # 数据流水线：输入 -> 隐藏层 -> 激活函数 -> 输出层
        x = self.hidden_layer(x)
        x = self.activation(x)
        output = self.output_layer(x)
        return output

def compute_pde_residual(model, x, t, alpha):
    T = model.forward(torch.cat([x, t], dim=1))
    dT_dt = torch.autograd.grad(
        outputs=T, inputs=t, 
        grad_outputs=torch.ones_like(T), 
        create_graph=True)[0]
    dT_dx = torch.autograd.grad(outputs=T, inputs=x, 
        grad_outputs=torch.ones_like(T),
        create_graph=True)[0]
    d2T_dx2 = torch.autograd.grad(outputs=dT_dx, inputs=x,
        grad_outputs=torch.ones_like(T),
        create_graph=True)[0]
    residual = dT_dt - alpha * d2T_dx2
    return residual

model = SimpleMLP()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

print("开始训练...")
for epoch in range(epochs):
    optimizer.zero_grad()

    residual = compute_pde_residual(model, x, t, alpha)
    loss_pde = F.mse_loss(residual, torch.zeros_like(residual))
    
    loss_ic = F.mse_loss(model(torch.cat([x_ic, t_ic], dim=1)), torch.sin(x_ic * math.pi))
    T_left = model(torch.cat([x_b, t_b], dim=1))
    T_right = model(torch.cat([x_c, t_c], dim=1))
    loss_bc = F.mse_loss(T_left, torch.zeros_like(T_left)) + F.mse_loss(T_right, torch.zeros_like(T_right))

    loss_total = loss_pde + loss_ic + loss_bc

    loss_total.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss Total: {loss_total.item():.6f} "
              f"(PDE: {loss_pde.item():.6f}, IC: {loss_ic.item():.6f}, BC: {loss_bc.item():.6f})")
print("训练完成！")

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

T_exact = np.sin(np.pi * X_np) * np.exp(-alpha_val * (np.pi**2) * T_np)
error_l2 = np.linalg.norm(T_pred - T_exact, 2) / np.linalg.norm(T_exact, 2)
print(f"相对 L2 误差 (Relative L2 Error): {error_l2:.4e}")

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
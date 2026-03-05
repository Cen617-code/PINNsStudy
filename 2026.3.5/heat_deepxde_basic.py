"""
DeepXDE 基础教学：求解完整的 1D 热传导方程 (固定 α)
====================================================
目标：熟悉 DeepXDE 的核心 API 架构 (Geometry, TimeDomain, IC/BC, TimePDE, FNN, Model)

方程：dT/dt = α * d²T/dx²
空间：x ∈ [0, 1]
时间：t ∈ [0, 1]
初始条件：T(x, 0) = sin(πx)
边界条件：T(0, t) = 0, T(1, t) = 0
热扩散系数：α = 0.05
"""

import deepxde as dde
import numpy as np

# 1. 定义物理常数
alpha = 0.05

# ==================== [引导问题 1：几何与时间域] ====================
# 在 Phase 3 中，我们用 torch.rand() 在 [0,1] 区间内生硬地采点。
# 在 DeepXDE 中，我们只需“声明”几何形状和时间跨度。
# 查阅官方文档：
# - 用 dde.geometry.Interval 定义 1D 空间 [0, 1]
# - 用 dde.geometry.TimeDomain 定义时间域 [0, 1]
# - 将它们组合成时空域 dde.geometry.GeometryXTime
geom = dde.geometry.Interval(0, 1)
timedomain = dde.geometry.TimeDomain(0, 1)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

# ==================== [引导问题 2：PDE 残差计算] ====================
# 这里是 DeepXDE 最强大的地方！再也不用写又臭又长的 autograd.grad 链了。
# 任务：参考热传导官方示例，用 dde.grad.jacobian 和 dde.grad.hessian 完成热传导方程残差。
# 输入参数：x (是一个 N×2 的矩阵，第一列 [:,0] 是位置，第二列 [:,1] 是时间), y (网络输出的温度 T)
def pde(x, y):
    # 用 dde.grad.jacobian 求 y 对 t (也就是 x[:, 1]) 的一阶导数：dT/dt
    # 用 dde.grad.hessian 求 y 对空间 (也就是 x[:, 0]) 的二阶导数：d²T/dx²
    # 提示：jacobian 的 i,j 参数表示 y 的第 i 列对 x 的第 j 列求导。这里 y 只有1列所以 i=0。
    
    dT_dt = dde.grad.jacobian(y, x, i=0, j=1) 
    d2T_dx2 = dde.grad.hessian(y, x, i=0, j=0)
    
    return dT_dt - alpha * d2T_dx2
    pass 

# ==================== [引导问题 3：边界条件 (BC) 与初始条件 (IC)] ====================
# DeepXDE 中，条件是通过函数定义的，而不是预先采好一堆点。
# on_boundary 和 on_initial 是 DeepXDE 内部判断一个点是否在边界上的布尔回调函数。

# 边界条件：T(0, t) = 0 和 T(1, t) = 0
# `lambda x: 0` 是目标值函数：不管输入什么点，约束它等于 0
bc = dde.icbc.DirichletBC(geomtime, lambda x: 0, lambda _, on_boundary: on_boundary)

# 初始条件：T(x, 0) = sin(πx)
# `lambda x: np.sin(np.pi * x[:, 0:1])` 提取出空间坐标求正弦
ic = dde.icbc.IC(geomtime, lambda x: np.sin(np.pi * x[:, 0:1]), lambda _, on_initial: on_initial)

# ==================== [结构组装与训练配置] ====================

# 把几何、方程、条件组合成一个 "偏微分方程实例"
# num_domain: 我们曾经的 N_f
# num_boundary: 我们曾经的 N_bc
# num_initial: 我们曾经的 N_ic
data = dde.data.TimePDE(
    geomtime, pde, [bc, ic], 
    num_domain=1000, num_boundary=100, num_initial=100
)

# 搭建网络 (和我们在 Phase 3 的 SimpleMLP 完全等效)
# [2] (x, t) -> [50, 50, 50] Tanh -> [1] (T)
# 这里的 "Glorot normal" 是深度学习非常标准的 Xavier 权重初始化方法
net = dde.nn.FNN([2] + [50] * 3 + [1], "tanh", "Glorot normal")

# 组装网络和数据
model = dde.Model(data, net)

# 编译！DeepXDE 的 API 非常类似 Keras
model.compile("adam", lr=1e-3)

# 训练并自动记录历史
print("开始训练 DeepXDE 基础模型...")
losshistory, train_state = model.train(iterations=5000)

# ==================== [阶段验证任务] ====================
# 1. 补全 pde(x, y) 函数中的导数计算
# 2. 取消上面 `model.train` 的注释
# 3. 在终端运行此脚本，观察终端输出的 Epoch/Loss 信息，是不是比手写看着清爽多了？

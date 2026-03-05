"""
DeepXDE 进阶教学：动态重采样与 L-BFGS 两阶段优化
=================================================
目标：学习 DeepXDE 中的 PDEPointResampler 回调机制，以及 Adam → L-BFGS 联合训练策略。
"""

import deepxde as dde
import numpy as np

alpha = 0.05
geom = dde.geometry.Interval(0, 1)
timedomain = dde.geometry.TimeDomain(0, 1)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

def pde(x, y):
    dT_dt = dde.grad.jacobian(y, x, i=0, j=1) 
    d2T_dx2 = dde.grad.hessian(y, x, i=0, j=0)
    return dT_dt - alpha * d2T_dx2

bc = dde.icbc.DirichletBC(geomtime, lambda x: 0, lambda _, on_boundary: on_boundary)
ic = dde.icbc.IC(geomtime, lambda x: np.sin(np.pi * x[:, 0:1]), lambda _, on_initial: on_initial)

data = dde.data.TimePDE(
    geomtime, pde, [bc, ic], 
    num_domain=1000, num_boundary=100, num_initial=100
)

net = dde.nn.FNN([2] + [50] * 3 + [1], "tanh", "Glorot normal")
model = dde.Model(data, net)

# ==================== [引导问题 4：动态重采样回调] ====================
# 在 Phase 3 中，我们在 for epoch: 循环里写 torch.rand() 实现重采样。
# DeepXDE 使用 回调函数 (Callback) 机制在训练过程中无缝插入操作。

# 任务：参考官方 resample 示例，定义一个 PDEPointResampler 回调，
# 让它每隔 100 轮重新在域内采样一次点。
pde_resampler = dde.callbacks.Callback
pde_resampler = None

# ==================== [阶段一：Adam 粗调] ====================
print("Stage 1: Adam 优化器粗调...")
model.compile("adam", lr=1e-3)
# 注意这里我们把回调传给 model.train
losshistory, train_state = model.train(
    iterations=5000, 
    callbacks=[pde_resampler] if pde_resampler else None
)

# ==================== [阶段二：L-BFGS 微调 (拟牛顿法)] ====================
# L-BFGS 是一个二阶优化器。它利用了海森矩阵的近似信息，能极快地找到局部最优，
# 但不能乱走，所以通常在 Adam 把模型带到最优解附近后再开 L-BFGS "收割"。
print("-" * 50)
print("Stage 2: L-BFGS 优化器精准微调...")

# 重新编译模型，这次优化器换成 "L-BFGS" (注意拼写)
# 不需要指定 lr，L-BFGS 自己会带线搜索 (line search) 找步长。
model.compile("L-BFGS")
losshistory, train_state = model.train(
    callbacks=[pde_resampler] if pde_resampler else None
)

# 再次调用 model.train()。
# 对 L-BFGS，不需要指定 iterations，它会在满足梯度容差时自动停下。
# 但别忘了把 pde_resampler 也要传进去！
# losshistory, train_state = model.train(???)

# ==================== [阶段验证任务] ====================
# 1. 补全 pde_resampler 定义 
# 2. 补全 Stage 2 中 model.compile 和 model.train 的调用
# 3. 运行这个新脚本，观察终端：Adam 跑完后，L-BFGS 是如何暴降 Loss 的！

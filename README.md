# PINNsStudy

这是一个围绕物理信息神经网络（PINNs）的分阶段学习与实验仓库。项目主线严格对应 [`Plan.md`](Plan.md) 中的四阶段路线，并把每一轮推导、实验现象和工程经验沉淀到 [`Note.md`](Note.md)。

当前仓库已经从 `PyTorch Autograd` 的高阶导数计算，推进到 1D 热传导方程、参数化 PINN、多 `alpha` 泛化验证、GPU 迁移，以及 `DeepXDE` 框架下的 1D/2D 热方程建模。它的直接目标，是为后续飞秒激光诱导表面形貌（LIPSS）轻量化代理模型打基础。

## 项目主线

| Phase | 核心主题 | 代表文件 | 当前状态 |
| --- | --- | --- | --- |
| Phase 1 | Autograd 与高阶导数 | `2026.2.22/mlp_basics.py` `2026.2.27/antograd_derivatives.py` | 已完成基础验证 |
| Phase 2 | 1D 热传导方程 PINN | `2026.2.27/heat_equation_pinn.py` `2026.2.28/weight_experiment.py` | 已完成基线训练与权重实验 |
| Phase 3 | 参数化 PINN 与代理模型 | `2026.2.28/parametric_pinn.py` `2026.3.1/multi_alpha_validation.py` `2026.3.1/multi_alpha_validation_GPU.py` | 已完成多 `alpha` 泛化与 GPU 迁移 |
| Phase 4 | DeepXDE 工业级框架 | `2026.3.5/heat_deepxde_basic.py` `2026.3.5/heat_deepxde_parametric.py` `2026.3.5/heat_deepxde_2d.py` | 已完成 1D、参数化和 2D 原型；`heat_deepxde_resample.py` 仍是教学练习骨架 |

## 当前仓库结构

下面的目录树省略了 `.git/` 和 `venv/` 这类元数据或本地环境目录，只保留当前学习主线直接相关的文件。

```text
PINNsStudy/
├── AGENTS.md                         # 仓库内协作与教学约束
├── Plan.md                           # PINNs 四阶段学习与开发计划
├── Note.md                           # 阶段性物理推导、API 笔记、实验结论
├── README.md                         # 项目总览
├── Test.md                           # 预留测试记录（当前为空）
├── requirements.txt                  # 当前环境依赖锁定文件
├── 2026.2.22/
│   └── mlp_basics.py                 # 最小 MLP 骨架
├── 2026.2.27/
│   ├── antograd_derivatives.py       # Autograd 一阶/二阶导数示例
│   ├── heat_equation_pinn.py         # 1D 热传导方程 PINN 基线
│   └── heat_equation_result.png      # Phase 2 结果图
├── 2026.2.28/
│   ├── heat_equation_result.png      # 阶段复现结果图
│   ├── parametric_pinn.py            # 参数化 1D 热传导 PINN
│   ├── parametric_pinn_result.png    # 参数化 PINN 结果图
│   └── weight_experiment.py          # Loss 权重平衡实验
├── 2026.3.1/
│   ├── multi_alpha_error.png         # 多 alpha 泛化误差曲线
│   ├── multi_alpha_validation.py     # 多 alpha 验证 + 数据损失 + 重采样
│   └── multi_alpha_validation_GPU.py # GPU 加速版本
└── 2026.3.5/
    ├── deepxde_multi_alpha_error.png # DeepXDE 参数化误差曲线
    ├── heat_2d_error.png             # DeepXDE 2D 热传导结果图
    ├── heat_deepxde_2d.py            # DeepXDE 2D 热传导方程
    ├── heat_deepxde_basic.py         # DeepXDE 1D 入门版
    ├── heat_deepxde_parametric.py    # DeepXDE 参数化 PINN
    └── heat_deepxde_resample.py      # DeepXDE 重采样教学骨架（待补全）
```

## 关键物理问题

### 1. 1D 热传导方程

```text
dT/dt = alpha * d²T/dx²
```

完整问题写成：

$$
\frac{\partial T(x,t)}{\partial t} = \alpha \frac{\partial^2 T(x,t)}{\partial x^2}, \quad x \in [0,1], \ t \in [0,1]
$$

$$
T(x,0) = \sin(\pi x)
$$

$$
T(0,t) = 0, \quad T(1,t) = 0
$$

- 空间域：`x in [0, 1]`
- 时间域：`t in [0, 1]`
- 初始条件：`T(x, 0) = sin(pi * x)`
- 边界条件：`T(0, t) = 0, T(1, t) = 0`

### 2. 参数化热传导方程

```text
输入: (x, t, alpha)
输出: T(x, t, alpha)
```

对应的参数化控制方程可以写成：

$$
\frac{\partial T(x,t,\alpha)}{\partial t} = \alpha \frac{\partial^2 T(x,t,\alpha)}{\partial x^2}, \quad x \in [0,1], \ t \in [0,1], \ \alpha \in [0.01, 0.1]
$$

$$
T(x,0,\alpha) = \sin(\pi x)
$$

$$
T(0,t,\alpha) = 0, \quad T(1,t,\alpha) = 0
$$

这里的核心思想不是“再解一个单独 PDE”，而是让网络一次性学习一族解，从而具备代理模型能力。

### 3. 2D 热传导方程

```text
dT/dt = alpha * (d²T/dx² + d²T/dy²)
```

完整形式为：

$$
\frac{\partial T(x,y,t)}{\partial t} = \alpha \left(\frac{\partial^2 T(x,y,t)}{\partial x^2} + \frac{\partial^2 T(x,y,t)}{\partial y^2}\right)
$$

这是从 1D 空间域扩展到 2D 空间域的进阶版本，用于验证高维几何、二维 Hessian 和 DeepXDE 回调机制。

## 环境依赖

仓库现在已经提供 [`requirements.txt`](requirements.txt)。按当前文件内容，核心依赖包括：

- `torch==2.10.0`
- `DeepXDE==1.15.0`
- `numpy==2.4.2`
- `matplotlib==3.10.8`
- `scipy==1.17.1`
- `scikit-learn==1.8.0`

本地 `venv/pyvenv.cfg` 显示当前实验环境使用的是 `Python 3.13.3`。如果你要重建环境，建议优先使用接近版本的 Python。

一个最小安装流程如下：

```bash
python3.13 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

说明：

- 手写 PINN 脚本依赖 `PyTorch`。
- `DeepXDE` 相关脚本依赖 `DeepXDE` 与对应后端；当前依赖文件已经包含 `torch`，因此这条学习线默认按 PyTorch 后端理解即可。
- `2026.3.1/multi_alpha_validation_GPU.py` 只有在本机具备可用 CUDA 与 CUDA 版 PyTorch 时，才会真正跑在 GPU 上。

## 建议阅读与运行顺序

建议先读文档，再按 Phase 顺序推进。由于多个脚本使用了相对路径保存图片，最好进入脚本所在目录运行，这样产出的图会直接落在对应日期文件夹中。

### 1. 先读文档

```bash
sed -n '1,220p' Plan.md
sed -n '1,320p' Note.md
```

### 2. 再按阶段运行

```bash
cd 2026.2.22
python mlp_basics.py

cd ../2026.2.27
python antograd_derivatives.py
python heat_equation_pinn.py

cd ../2026.2.28
python weight_experiment.py
python parametric_pinn.py

cd ../2026.3.1
python multi_alpha_validation.py
python multi_alpha_validation_GPU.py

cd ../2026.3.5
python heat_deepxde_basic.py
python heat_deepxde_parametric.py
python heat_deepxde_2d.py
```

`2026.3.5/heat_deepxde_resample.py` 当前仍保留为引导式练习，不建议把它作为默认直跑脚本。

## 脚本状态说明

| 文件 | 角色 | 状态 |
| --- | --- | --- |
| `2026.2.22/mlp_basics.py` | Phase 1 入门骨架 | 可直接阅读/运行 |
| `2026.2.27/antograd_derivatives.py` | 输入导数练习 | 可直接运行 |
| `2026.2.27/heat_equation_pinn.py` | 1D 热传导基线实验 | 可直接运行 |
| `2026.2.28/weight_experiment.py` | Loss 权重平衡实验 | 可直接运行 |
| `2026.2.28/parametric_pinn.py` | 参数化 PINN 基础版 | 可直接运行 |
| `2026.3.1/multi_alpha_validation.py` | 多 `alpha` 泛化验证 | 可直接运行 |
| `2026.3.1/multi_alpha_validation_GPU.py` | GPU 迁移版 | 可直接运行，需 GPU 才有加速效果 |
| `2026.3.5/heat_deepxde_basic.py` | DeepXDE 1D 入门脚本 | 可直接运行，保留了教学式注释 |
| `2026.3.5/heat_deepxde_parametric.py` | DeepXDE 参数化版本 | 可直接运行，保留了教学式注释 |
| `2026.3.5/heat_deepxde_2d.py` | DeepXDE 2D 版本 | 可直接运行 |
| `2026.3.5/heat_deepxde_resample.py` | DeepXDE 回调练习 | 当前含占位逻辑，用于练习 `PDEPointResampler` 与双阶段优化 |

## 阶段性结果

根据 [`Note.md`](Note.md) 中已经记录的实验结论，这个仓库当前完成了以下几个关键里程碑：

- Phase 2 中，1D 热传导方程 PINN 已稳定收敛，相对 L2 误差达到 `1.14e-03` 量级。
- Loss 权重实验表明，在这个基础问题上等权组合最稳，盲目放大某一项损失会显著拉差整体误差。
- Phase 3 中，参数化 PINN 已具备跨 `alpha in [0.01, 0.1]` 的泛化能力，并观察到了典型的 U 型误差分布。
- Phase 3 中，GPU 迁移版本已经完成设备一致性改造，为后续更大规模训练做了准备。
- Phase 4 中，项目已经迁移到 `DeepXDE`，并扩展到了 2D 热传导方程；笔记中记录的 2D 相对 L2 误差达到 `5.88e-04`。

## 结果图集

### Phase 2: 1D 热传导方程基线结果

`2026.2.27/heat_equation_pinn.py`

![Phase 2 1D 热传导方程结果](2026.2.27/heat_equation_result.png)

### Phase 2: 阶段复现结果图

`2026.2.28/heat_equation_result.png`

![Phase 2 阶段复现图](2026.2.28/heat_equation_result.png)

### Phase 3: 参数化 PINN 结果

`2026.2.28/parametric_pinn_result.png`

![Phase 3 参数化 PINN 结果](2026.2.28/parametric_pinn_result.png)

### Phase 3: 多 alpha 泛化误差曲线

`2026.3.1/multi_alpha_error.png`

![Phase 3 多 alpha 泛化误差](2026.3.1/multi_alpha_error.png)

### Phase 4: DeepXDE 参数化 PINN 泛化误差

`2026.3.5/deepxde_multi_alpha_error.png`

![Phase 4 DeepXDE 多 alpha 泛化误差](2026.3.5/deepxde_multi_alpha_error.png)

### Phase 4: DeepXDE 2D 热传导方程结果

`2026.3.5/heat_2d_error.png`

![Phase 4 DeepXDE 2D 热传导结果](2026.3.5/heat_2d_error.png)

## 文档关系

- [`Plan.md`](Plan.md)：回答“接下来按什么阶段推进”
- [`Note.md`](Note.md)：回答“已经学会了什么、踩过哪些坑”
- [`README.md`](README.md)：回答“这个仓库当前长什么样、怎么运行、看哪些结果”
- [`Test.md`](Test.md)：预留测试记录入口，当前还是空文件

## 下一步方向

从当前结构看，热传导方程这条训练链已经基本跑通。后续可以继续沿着下面几条线推进：

- 将 `alpha` 扩展为更多材料或工艺参数，逐步逼近 LIPSS 代理模型的真实输入空间
- 从解析解监督过渡到 COMSOL 或实验数据驱动
- 在 `PyTorch` 与 `DeepXDE` 两条路线中继续尝试更复杂的多物理场残差构建

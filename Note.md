# PINNs 学习笔记

## Phase 1: 掌握 Autograd 与时空导数
**日期**: 2026.2.22

### 1. 物理场逼近器的基本骨架 (SimpleMLP)
- **设计思路**：在 PINNs 中，神经网络被视作连续物理场的通用逼近器。输入一般是时空坐标 $(x, t)$，输出是待求解的物理量（如温度 $T$、形貌深度），模型每一层都在进行空间的非线性映射。
- **PyTorch 实现**：通过继承 `torch.nn.Module`，利用 `nn.Linear` 搭建全连接隐藏层，并在 `forward` 方法里规定连续时空坐标张量流动的计算拓扑。

### 2. 激活函数的物理意义 (至关重要)
- **物理陷阱：为什么 PINNs 绝不能用 ReLU？** 
  ReLU 的数学表达为 $f(x) = \max(0, x)$。它的一阶导数是分段常数（0 或 1），更致命的是，它的**二阶偏导数几乎处处为 0**！
  在很多真实物理方程中（如含 $\frac{\partial^2 T}{\partial x^2}$ 的热传导方程），如果激活函数用 ReLU，网络算出的二阶偏导数将永远是 0。物理残差损失（PDE Loss）会失去对网络权重的指导能力，相当于“计算图在物理层面上断裂了”。
- **工程结论**：在 PINNs 领域计算偏微分方程残差时，必须使用具备二阶及以上平滑连续可导性质的激活函数，如 `Tanh`、`Sin` 或 `Sigmoid`。

### 3. 用 `torch.autograd.grad` 计算输入导数 (PINNs 运算引擎)
**日期**: 2026.2.27

- **核心区分**：优化器对**网络权重**求梯度（用于参数更新），而 PINNs 需要对**输入坐标** `(x, t)` 求梯度（用于构建物理残差）。后者正是 `torch.autograd.grad` 的职责。
- **关键 API 用法**：
  ```python
  dT_dx = torch.autograd.grad(outputs=T, inputs=x,
                               grad_outputs=torch.ones_like(T),
                               create_graph=True)[0]
  ```
  - `outputs`：要对谁的输出求导（网络预测值 `T`）
  - `inputs`：对哪个输入变量求导（坐标 `x`）
  - `grad_outputs`：上游种子梯度，shape 需与 `outputs` 一致，一般设为 `torch.ones_like(T)`
  - `create_graph=True`：**至关重要！** 让导数结果本身也挂载在计算图上，从而支持高阶求导
  - `[0]`：返回值是元组，真正的导数张量在第 0 个位置

- **高阶导数链**：
  对一阶导数 `dT_dx` 再次调用 `autograd.grad`，`inputs` 仍然传入 `x`，即可得到二阶导数 `d²T/dx²`（拉普拉斯算子项），完整链路为：
  `输入 x → 网络 T → dT/dx → d²T/dx²`

- **计算图连续性验证**：输出张量中出现 `grad_fn=<SliceBackward0>` 即证明计算图未断裂，导数值仍然可以参与后续反向传播。

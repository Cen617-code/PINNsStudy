import torch
import torch.nn as nn

class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        # 定义输入层到隐藏层（输入维度2 -> 隐藏神经元20）
        self.hidden_layer = nn.Linear(2, 20)
        # 定义激活函数
        self.activation = nn.Tanh()
        # 定义隐藏层到输出层（隐藏神经元20 -> 输出维度1）
        self.output_layer = nn.Linear(20, 1)

    def forward(self, x):
        # 数据流水线：输入 -> 隐藏层 -> 激活函数 -> 输出层
        x = self.hidden_layer(x)
        x = self.activation(x)
        output = self.output_layer(x)
        return output

# 测试一下我们的网络
if __name__ == "__main__":
    # 创建网络实例
    model = SimpleMLP()
    # 随机生成一个批次的坐标点 (Batch_size=5, 维度=2)
    dummy_input = torch.randn(5, 2) 
    # 计算预测值
    prediction = model(dummy_input)
    print("模型输出形状:", prediction.shape) # 期望输出形状为 [5, 1]

import torch
import torch.nn as nn
import torch.nn.functional as F


class Pure1DCNN(nn.Module):
    """
    轻量级1D-CNN模型（LSTM消融实验）
    参考论文思路：卷积核16，移除LSTM，总参数量约2.5K
    """

    def __init__(self, input_dim=1, seq_length=10, conv_channels=16, dropout_rate=0.2):
        super(Pure1DCNN, self).__init__()

        # 1D-CNN模块
        self.conv1d = nn.Conv1d(
            in_channels=input_dim,
            out_channels=conv_channels,
            kernel_size=3,
            padding=1
        )
        self.batchnorm = nn.BatchNorm1d(conv_channels)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        # 全局平均池化
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)

        # 输出层
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(conv_channels, 1)

        # 参数初始化
        self._initialize_weights()

    def _initialize_weights(self):
        """Xavier均匀初始化"""
        for module in self.modules():
            if isinstance(module, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x):
        # x形状: (batch_size, seq_length, input_dim)

        # 转换维度为(batch_size, input_dim, seq_length)
        x = x.transpose(1, 2)

        # 1D-CNN
        x = self.conv1d(x)
        x = F.relu(x)
        x = self.batchnorm(x)
        x = self.pool(x)  # 输出形状: (batch, conv_channels, seq_len//2)

        # 全局平均池化
        x = self.global_avg_pool(x)  # (batch, conv_channels, 1)
        x = x.squeeze(-1)  # (batch, conv_channels)

        # 输出层
        x = self.dropout(x)
        x = self.fc(x)
        x = torch.sigmoid(x)

        return x

    def get_parameter_count(self):
        """返回模型参数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_model(config=None):
    """创建模型实例"""
    if config is None:
        config = {
            'input_dim': 1,
            'seq_length': 10,
            'conv_channels': 16,
            'dropout_rate': 0.2
        }

    return Pure1DCNN(**config)


if __name__ == "__main__":
    # 测试模型
    model = create_model()
    print(f"模型总参数量: {model.get_parameter_count():,}")

    # 测试前向传播
    test_input = torch.randn(32, 10, 1)  # batch_size=32, seq_length=10, input_dim=1
    output = model(test_input)
    print(f"输入形状: {test_input.shape}")
    print(f"输出形状: {output.shape}")
    print(f"输出范围: [{output.min().item():.4f}, {output.max().item():.4f}]")
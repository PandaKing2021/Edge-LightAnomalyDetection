import torch
import torch.nn as nn
import torch.nn.functional as F


class PureLSTM(nn.Module):
    """
    纯LSTM模型（消融实验：移除CNN）
    参考论文思路：LSTM隐藏单元32，总参数量约4.2K
    """

    def __init__(self, input_dim=1, seq_length=10,
                 lstm_hidden=32, lstm_layers=2, dropout_rate=0.2):
        super(PureLSTM, self).__init__()
        
        # LSTM模块
        self.lstm = nn.LSTM(
            input_size=input_dim,  # 直接使用原始输入维度
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout_rate if lstm_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout_rate)
        
        # 输出层
        self.fc = nn.Linear(lstm_hidden, 1)
        
        # 参数初始化
        self._initialize_weights()

    def _initialize_weights(self):
        """Xavier均匀初始化"""
        for module in self.modules():
            if isinstance(module, (nn.Linear)):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_uniform_(param)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0)

    def forward(self, x):
        # x形状: (batch_size, seq_length, input_dim)
        # 直接输入LSTM，无需维度转换
        
        # LSTM
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # 取最后一个时间步的输出
        x = lstm_out[:, -1, :]
        x = self.dropout(x)
        
        # 输出层
        x = self.fc(x)
        x = torch.sigmoid(x)
        
        return x

    def get_parameter_count(self):
        """返回模型参数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_model(config=None):
    """创建仅LSTM模型实例"""
    if config is None:
        config = {
            'input_dim': 1,
            'seq_length': 10,
            'lstm_hidden': 32,
            'lstm_layers': 2,
            'dropout_rate': 0.2
        }
    
    return PureLSTM(**config)


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
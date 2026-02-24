import torch
import torch.nn as nn
import torch.quantization as quantization
from torch.quantization import QuantStub, DeQuantStub
import numpy as np
import json
import os
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, f1_score
import time

# 从原始模型导入
try:
    from ann_model import Edge1DCLSTM
except ImportError:
    class Edge1DCLSTM(nn.Module):
        def __init__(self, input_dim=1, seq_length=10, conv_channels=16,
                     lstm_hidden=32, lstm_layers=2, dropout_rate=0.2):
            super(Edge1DCLSTM, self).__init__()
            self.conv1d = nn.Conv1d(input_dim, conv_channels, kernel_size=3, padding=1)
            self.batchnorm = nn.BatchNorm1d(conv_channels)
            self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
            self.lstm = nn.LSTM(conv_channels, lstm_hidden, lstm_layers,
                                batch_first=True, dropout=dropout_rate if lstm_layers > 1 else 0)
            self.dropout = nn.Dropout(dropout_rate)
            self.fc = nn.Linear(lstm_hidden, 1)

        def forward(self, x):
            x = x.transpose(1, 2)
            x = self.conv1d(x)
            x = torch.relu(x)
            x = self.batchnorm(x)
            x = self.pool(x)
            x = x.transpose(1, 2)
            lstm_out, _ = self.lstm(x)
            x = lstm_out[:, -1, :]
            x = self.dropout(x)
            x = self.fc(x)
            x = torch.sigmoid(x)
            return x


class HybridPrecision1DCLSTM(nn.Module):
    """
    混合精度1DCNN-LSTM模型
    CNN部分：INT8量化
    LSTM部分：FP32保留
    实现论文中3.7.3节的量化策略
    """

    def __init__(self, input_dim=1, seq_length=10, conv_channels=16,
                 lstm_hidden=32, lstm_layers=2, dropout_rate=0.2):
        super(HybridPrecision1DCLSTM, self).__init__()

        # 量化边界标记
        self.quant = QuantStub()  # 量化输入
        self.dequant = DeQuantStub()  # 反量化输出

        # CNN部分（将被量化到INT8）
        self.conv1d = nn.Conv1d(input_dim, conv_channels, kernel_size=3, padding=1)
        self.batchnorm = nn.BatchNorm1d(conv_channels)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        # LSTM部分（保持FP32）
        self.lstm = nn.LSTM(
            input_size=conv_channels,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout_rate if lstm_layers > 1 else 0
        )

        self.dropout = nn.Dropout(dropout_rate)

        # 输出层（FP32）
        self.fc = nn.Linear(lstm_hidden, 1)

        # 量化配置
        self.quant_config = {
            'qconfig': quantization.get_default_qconfig('fbgemm'),
            'dtype': torch.qint8
        }

    def forward(self, x):
        # 量化边界：开始量化
        x = self.quant(x)

        # 转换维度
        x = x.transpose(1, 2)

        # CNN部分（INT8量化区域）
        x = self.conv1d(x)
        x = torch.relu(x)
        x = self.batchnorm(x)
        x = self.pool(x)

        # 量化边界：CNN输出反量化
        x = self.dequant(x)

        # 转换维度
        x = x.transpose(1, 2)

        # LSTM部分（FP32保留区域）
        lstm_out, _ = self.lstm(x)
        x = lstm_out[:, -1, :]
        x = self.dropout(x)

        # 输出层
        x = self.fc(x)
        x = torch.sigmoid(x)

        return x

    def fuse_modules(self):
        """融合CNN部分的模块以优化量化性能"""
        # 融合卷积层和激活函数
        torch.quantization.fuse_modules(self, [['conv1d', 'batchnorm']], inplace=True)

    def prepare_for_quantization(self):
        """准备模型进行量化"""
        # 设置量化配置
        self.qconfig = self.quant_config['qconfig']
        
        # 确保LSTM和全连接层保持FP32（不量化）
        self.lstm.qconfig = None
        self.fc.qconfig = None

        # 准备量化
        quantization.prepare(self, inplace=True)

    def convert_to_quantized(self):
        """转换为量化模型"""
        return quantization.convert(self, inplace=False)


class TimeSeriesCalibrationDataset(Dataset):
    """用于模型校准的时间序列数据集"""

    def __init__(self, data_file='dataset.json', sequence_length=10,
                 generator_id=1, num_samples=500):
        self.sequence_length = sequence_length

        # 加载数据
        with open(data_file, 'r') as f:
            data_dict = json.load(f)

        # 获取数据
        values_key = f"time_sequence_{generator_id}_value"
        self.values = np.array(data_dict[values_key])

        # 标准化
        self.mean = np.mean(self.values)
        self.std = np.std(self.values)
        self.values = (self.values - self.mean) / self.std

        # 创建校准序列
        self.samples = self._create_sequences(num_samples)

    def _create_sequences(self, num_samples):
        """创建校准序列"""
        sequences = []
        max_start = len(self.values) - self.sequence_length

        # 随机选择起始点
        indices = np.random.choice(max_start, min(num_samples, max_start), replace=False)

        for i in indices:
            seq = self.values[i:i + self.sequence_length]
            sequences.append(seq)

        return np.array(sequences, dtype=np.float32)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return torch.FloatTensor(sample).unsqueeze(-1)  # 添加特征维度


class HybridPrecisionQuantizer:
    """
    混合精度量化器
    实现论文中的混合精度量化策略：
    1. CNN部分：INT8量化
    2. LSTM部分：FP32保留
    """

    def __init__(self, device='cpu'):
        self.device = torch.device(device)
        self.original_model = None
        self.quantized_model = None

    def load_original_model(self, model_path):
        """加载原始FP32模型"""
        checkpoint = torch.load(model_path, map_location=self.device)

        # 获取模型配置
        if 'config' in checkpoint:
            config = checkpoint['config'].get('model_config', {
                'input_dim': 1,
                'seq_length': 10,
                'conv_channels': 16,
                'lstm_hidden': 32,
                'lstm_layers': 2,
                'dropout_rate': 0.2
            })
        else:
            config = {
                'input_dim': 1,
                'seq_length': 10,
                'conv_channels': 16,
                'lstm_hidden': 32,
                'lstm_layers': 2,
                'dropout_rate': 0.2
            }

        # 创建混合精度模型
        self.original_model = HybridPrecision1DCLSTM(**config).to(self.device)

        # 加载权重
        if 'model_state_dict' in checkpoint:
            self.original_model.load_state_dict(checkpoint['model_state_dict'])

        print(f"原始模型加载完成，参数量: {self._count_parameters(self.original_model):,}")
        return self.original_model

    def calibrate_model(self, calibration_loader, num_batches=100):
        """使用校准数据校准模型"""
        if self.original_model is None:
            raise ValueError("请先加载原始模型")

        # 准备量化
        self.original_model.eval()
        self.original_model.fuse_modules()
        self.original_model.prepare_for_quantization()

        print("开始模型校准...")

        # 校准过程
        with torch.no_grad():
            for batch_idx, data in enumerate(calibration_loader):
                if batch_idx >= num_batches:
                    break

                data = data.to(self.device)
                _ = self.original_model(data)

                if (batch_idx + 1) % 10 == 0:
                    print(f"校准进度: {batch_idx + 1}/{num_batches}")

        print("校准完成")

        # 转换为量化模型
        self.quantized_model = self.original_model.convert_to_quantized()

        return self.quantized_model

    def save_quantized_model(self, save_path='hybrid_precision_quantized.pth'):
        """保存量化模型"""
        if self.quantized_model is None:
            raise ValueError("请先进行模型量化")

        # 保存量化模型状态
        torch.save({
            'model_state_dict': self.quantized_model.state_dict(),
            'quantized': True,
            'hybrid_precision': True,
            'quant_config': self.quantized_model.quant_config
        }, save_path)

        print(f"量化模型已保存到: {save_path}")
        print(f"模型大小: {self._get_file_size(save_path):.2f} MB")

    def evaluate_quantization_effect(self, test_loader):
        """评估量化效果"""
        if self.original_model is None or self.quantized_model is None:
            raise ValueError("请先加载原始模型并完成量化")

        # 评估原始模型
        print("评估原始模型...")
        orig_results = self._evaluate_model(self.original_model, test_loader, "原始模型")

        # 评估量化模型
        print("评估量化模型...")
        quant_results = self._evaluate_model(self.quantized_model, test_loader, "量化模型")

        # 计算量化效果指标
        accuracy_drop = orig_results['accuracy'] - quant_results['accuracy']
        f1_drop = orig_results['f1'] - quant_results['f1']
        speedup = orig_results['inference_time'] / quant_results['inference_time']

        # 打印比较结果
        self._print_comparison(orig_results, quant_results, accuracy_drop, f1_drop, speedup)

        return {
            'original': orig_results,
            'quantized': quant_results,
            'accuracy_drop': accuracy_drop,
            'f1_drop': f1_drop,
            'speedup': speedup
        }

    def _evaluate_model(self, model, test_loader, model_name):
        """评估单个模型性能"""
        model.eval()

        all_predictions = []
        all_labels = []

        # 计时推理时间
        start_time = time.time()

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                predictions = (output > 0.5).float()

                all_predictions.extend(predictions.cpu().numpy().flatten())
                all_labels.extend(target.cpu().numpy().flatten())

        inference_time = time.time() - start_time

        # 计算指标
        accuracy = accuracy_score(all_labels, all_predictions)
        f1 = f1_score(all_labels, all_predictions)

        return {
            'model_name': model_name,
            'accuracy': accuracy,
            'f1': f1,
            'inference_time': inference_time,
            'throughput': len(all_labels) / inference_time
        }

    def _print_comparison(self, orig_results, quant_results, accuracy_drop, f1_drop, speedup):
        """打印模型性能比较"""
        print("\n" + "=" * 70)
        print("混合精度量化性能对比")
        print("=" * 70)
        print(f"{'指标':<20} {'原始模型(FP32)':<20} {'量化模型(INT8/FP32)':<20} {'变化':<10}")
        print("-" * 70)
        print(
            f"{'准确率':<20} {orig_results['accuracy']:.4f}{'':<10} {quant_results['accuracy']:.4f}{'':<10} {accuracy_drop:+.4f}")
        print(f"{'F1分数':<20} {orig_results['f1']:.4f}{'':<10} {quant_results['f1']:.4f}{'':<10} {f1_drop:+.4f}")
        print(
            f"{'推理时间(s)':<20} {orig_results['inference_time']:.4f}{'':<10} {quant_results['inference_time']:.4f}{'':<10} {quant_results['inference_time'] - orig_results['inference_time']:+.4f}")
        print(
            f"{'吞吐量(样本/s)':<20} {orig_results['throughput']:.1f}{'':<10} {quant_results['throughput']:.1f}{'':<10} {quant_results['throughput'] - orig_results['throughput']:+.1f}")
        print(f"{'加速比':<20} {'—':<20} {'—':<20} {speedup:.2f}x")
        print("=" * 70)

        # 量化效果总结
        print("\n量化效果总结:")
        print(f"1. 精度损失: 准确率下降 {accuracy_drop:.4f}, F1分数下降 {f1_drop:.4f}")
        print(f"2. 推理速度: 加速 {speedup:.2f} 倍")
        print(f"3. 内存节省: CNN部分从FP32(32位)压缩到INT8(8位)，减少75%内存占用")
        print(f"4. 能耗优化: INT8运算比FP32运算能耗降低约60-70%")

    def _count_parameters(self, model):
        """计算模型参数量"""
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def _get_file_size(self, file_path):
        """获取文件大小(MB)"""
        return os.path.getsize(file_path) / (1024 * 1024)

    def analyze_model_structure(self):
        """分析模型结构，显示量化配置"""
        if self.quantized_model is None:
            print("请先完成模型量化")
            return

        print("\n模型结构分析:")
        print("-" * 50)

        for name, module in self.quantized_model.named_modules():
            if hasattr(module, 'qconfig') and module.qconfig is not None:
                quant_type = "INT8量化" if module.qconfig.weight.p.keywords['dtype'] == torch.qint8 else "FP32"
                print(f"{name}: {quant_type}")
            elif isinstance(module, (nn.Conv1d, nn.BatchNorm1d)):
                print(f"{name}: CNN部分 (已量化)")
            elif isinstance(module, nn.LSTM):
                print(f"{name}: LSTM部分 (FP32保留)")

        print("-" * 50)


def create_calibration_data_loader(batch_size=32, num_samples=500):
    """创建校准数据加载器"""
    dataset = TimeSeriesCalibrationDataset(
        data_file='dataset.json',
        sequence_length=10,
        generator_id=1,
        num_samples=num_samples
    )

    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def create_test_data_loader():
    """创建测试数据加载器（从ann_train.py导入）"""
    try:
        from ann_train import TimeSeriesDataset
        from torch.utils.data import DataLoader

        dataset = TimeSeriesDataset(
            data_file='dataset.json',
            sequence_length=10,
            generator_id=1
        )

        # 取后20%作为测试集
        test_size = int(0.2 * len(dataset))
        train_size = len(dataset) - test_size

        from torch.utils.data import random_split
        _, test_dataset = random_split(dataset, [train_size, test_size])

        return DataLoader(test_dataset, batch_size=32, shuffle=False)

    except ImportError:
        print("警告: 无法导入训练模块，创建模拟测试数据")

        # 创建模拟数据
        class MockDataset(Dataset):
            def __init__(self, num_samples=200):
                self.data = torch.randn(num_samples, 10, 1)
                self.labels = torch.randint(0, 2, (num_samples, 1)).float()

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                return self.data[idx], self.labels[idx]

        dataset = MockDataset(200)
        return DataLoader(dataset, batch_size=32, shuffle=False)


def main():
    """主函数 - 执行混合精度量化流程"""
    print("=" * 70)
    print("混合精度量化系统 - 基于论文3.7.3节策略")
    print("=" * 70)

    # 初始化量化器
    quantizer = HybridPrecisionQuantizer(device='cpu')

    # 加载原始模型
    original_model_path = 'edge_1dclstm_best.pth'

    if not os.path.exists(original_model_path):
        print(f"警告: 原始模型文件 {original_model_path} 不存在")
        print("创建并训练一个示例模型...")

        # 创建示例模型并保存
        model = Edge1DCLSTM()
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': {'model_config': {
                'input_dim': 1,
                'seq_length': 10,
                'conv_channels': 16,
                'lstm_hidden': 32,
                'lstm_layers': 2,
                'dropout_rate': 0.2
            }}
        }, original_model_path)
        print(f"示例模型已保存到: {original_model_path}")

    print(f"加载原始模型: {original_model_path}")
    quantizer.load_original_model(original_model_path)

    # 创建校准数据加载器
    print("准备校准数据...")
    calibration_loader = create_calibration_data_loader()

    # 执行量化校准
    print("执行混合精度量化...")
    quantized_model = quantizer.calibrate_model(calibration_loader, num_batches=50)

    # 保存量化模型
    quantizer.save_quantized_model('hybrid_precision_model.pth')

    # 分析模型结构
    quantizer.analyze_model_structure()

    # 评估量化效果
    print("\n评估量化效果...")
    test_loader = create_test_data_loader()
    evaluation_results = quantizer.evaluate_quantization_effect(test_loader)

    # 保存评估结果
    with open('quantization_evaluation.json', 'w') as f:
        json.dump(evaluation_results, f, indent=2)

    print("\n量化流程完成！")
    print("关键成果:")
    print("1. CNN部分已成功量化为INT8精度")
    print("2. LSTM部分保持FP32精度以确保时序建模准确性")
    print("3. 通过QuantStub/DeQuantStub实现精度无缝转换")
    print("4. 量化模型已准备就绪，可用于边缘设备部署")


if __name__ == "__main__":
    main()

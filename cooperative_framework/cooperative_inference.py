"""
协同推理框架核心模块
Cooperative Inference Framework Core Module

实现端-边协同分层推理框架
基于论文第2.4节设计：终端哨兵模型 + 边缘主模型
"""

import torch
import torch.nn as nn
import numpy as np
import json
import time
import os
from typing import Dict, Any, Tuple, Optional, List
from dataclasses import dataclass, field
from datetime import datetime

from .config import CooperativeConfig, PowerConfig, TimeConfig, get_default_config
from .energy_monitor import EnergyMonitor, EnergyStatistics


@dataclass
class InferenceResult:
    """推理结果"""
    # 预测分数 (0-1)
    score: float
    
    # 是否为异常
    is_anomaly: bool
    
    # 是否唤醒主模型
    wakeup_triggered: bool
    
    # 最终预测来源: 'sentinel' 或 'full'
    prediction_source: str
    
    # 推理时间 (s)
    inference_time: float
    
    # 能耗 (J)
    energy_consumed: float
    
    # 哨兵模型分数
    sentinel_score: Optional[float] = None
    
    # 主模型分数（如果唤醒）
    full_model_score: Optional[float] = None


class CooperativeInferenceFramework:
    """
    端-边协同推理框架
    
    核心功能：
    1. 终端哨兵模型持续监控（LSTM，参数量8,481）
    2. 边缘主模型按需唤醒（Edge-1DCNN-LSTM，参数量12,385）
    3. 动态阈值调整
    4. 能耗优化
    5. 推理结果统计
    
    设计目标（阈值=0.9时）：
    - F1分数: 0.9944
    - 能耗节省: 81.96%
    - 唤醒率: 8.55%
    """
    
    def __init__(
        self,
        sentinel_model: nn.Module,
        edge_model: nn.Module,
        config: Optional[CooperativeConfig] = None,
        device: str = 'cpu'
    ):
        """
        初始化协同推理框架
        
        Args:
            sentinel_model: 哨兵模型（轻量级LSTM）
            edge_model: 边缘主模型（Edge-1DCNN-LSTM）
            config: 配置参数
            device: 运行设备
        """
        self.config = config if config else get_default_config()
        self.device = torch.device(device)
        
        # 加载模型
        self.sentinel_model = sentinel_model.to(self.device)
        self.edge_model = edge_model.to(self.device)
        
        # 设置为评估模式
        self.sentinel_model.eval()
        self.edge_model.eval()
        
        # 能耗监控器
        self.energy_monitor = EnergyMonitor(
            power_config=self.config.power,
            time_config=self.config.time
        )
        
        # 推理统计
        self.inference_count = 0
        self.wakeup_count = 0
        self.total_inference_time = 0.0
        
        # 预测结果记录
        self.predictions: List[Dict] = []
        
        # 当前阈值
        self.threshold = self.config.threshold
    
    def inference(
        self,
        data: torch.Tensor,
        return_details: bool = False
    ) -> Tuple[float, Dict]:
        """
        执行协同推理
        
        Args:
            data: 输入数据，形状 (batch_size, seq_length, input_dim)
            return_details: 是否返回详细信息
            
        Returns:
            Tuple[float, Dict]: (预测分数, 详细信息)
        """
        start_time = time.time()
        
        with torch.no_grad():
            # Step 1: 终端哨兵模型推理
            sentinel_start = time.time()
            sentinel_output = self.sentinel_model(data)
            sentinel_score = sentinel_output.item() if sentinel_output.dim() == 0 else sentinel_output.squeeze().tolist()
            sentinel_time = time.time() - sentinel_start
            
            # Step 2: 判断是否需要唤醒主模型
            wakeup_triggered = False
            full_model_score = None
            
            # 哨兵模型分数超过阈值，或检测为异常（分数>0.5）时考虑唤醒
            if sentinel_score >= self.threshold:
                wakeup_triggered = True
                
                # Step 3: 唤醒边缘主模型
                full_start = time.time()
                full_output = self.edge_model(data)
                full_model_score = full_output.item() if full_output.dim() == 0 else full_output.squeeze().tolist()
                full_time = time.time() - full_start
                
                # 主模型结果作为最终结果
                final_score = full_model_score
                prediction_source = 'full'
            else:
                # 哨兵模型结果作为最终结果
                final_score = sentinel_score
                prediction_source = 'sentinel'
                full_time = 0
            
            total_time = time.time() - start_time
            
        # 记录能耗
        self.energy_monitor.record_inference(
            is_wakeup=wakeup_triggered,
            sentinel_time=sentinel_time,
            full_time=full_time if wakeup_triggered else None
        )
        
        # 更新统计
        self.inference_count += 1
        if wakeup_triggered:
            self.wakeup_count += 1
        self.total_inference_time += total_time
        
        # 计算能耗
        stats = self.energy_monitor.get_realtime_stats()
        energy_consumed = stats['total_energy'] / self.inference_count if self.inference_count > 0 else 0
        
        # 构建结果
        details = {
            'wakeup_triggered': wakeup_triggered,
            'prediction_source': prediction_source,
            'inference_time': total_time,
            'energy_consumed': energy_consumed,
            'sentinel_score': sentinel_score,
            'full_model_score': full_model_score
        }
        
        if return_details:
            return final_score, details
        else:
            return final_score, {'wakeup_triggered': wakeup_triggered}
    
    def batch_inference(
        self,
        data_loader,
        labels: Optional[List[int]] = None,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        批量推理
        
        Args:
            data_loader: 数据加载器
            labels: 真实标签（可选）
            verbose: 是否打印进度
            
        Returns:
            推理结果字典
        """
        self.energy_monitor.start_monitoring()
        
        all_predictions = []
        all_labels = []
        all_scores = []
        
        for batch_idx, batch_data in enumerate(data_loader):
            if isinstance(batch_data, (list, tuple)):
                data = batch_data[0]
                label = batch_data[1] if len(batch_data) > 1 else None
            else:
                data = batch_data
                label = None
            
            # 执行推理
            score, details = self.inference(data, return_details=True)
            
            # 收集结果
            if isinstance(score, list):
                all_scores.extend(score)
                all_predictions.extend([1 if s > 0.5 else 0 for s in score])
            else:
                all_scores.append(score)
                all_predictions.append(1 if score > 0.5 else 0)
            
            if label is not None:
                if isinstance(label, torch.Tensor):
                    all_labels.extend(label.cpu().numpy().flatten().tolist())
                else:
                    all_labels.extend(label.tolist() if hasattr(label, 'tolist') else label)
            
            if verbose and (batch_idx + 1) % 10 == 0:
                stats = self.energy_monitor.get_realtime_stats()
                print(f"处理进度: {batch_idx + 1}, 唤醒率: {stats['wakeup_rate']:.2%}, "
                      f"能耗节省: {stats['energy_saving']:.2f}%")
        
        # 停止监控
        stats = self.energy_monitor.stop_monitoring()
        
        # 计算评估指标
        results = {
            'predictions': all_predictions,
            'scores': all_scores,
            'labels': all_labels,
            'inference_count': self.inference_count,
            'wakeup_count': self.wakeup_count,
            'wakeup_rate': stats.wakeup_rate,
            'energy_saving': stats.energy_saving,
            'total_energy': stats.total_energy,
            'total_time': self.total_inference_time
        }
        
        # 如果有标签，计算评估指标
        if all_labels:
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            
            results.update({
                'accuracy': accuracy_score(all_labels, all_predictions),
                'precision': precision_score(all_labels, all_predictions, zero_division=0),
                'recall': recall_score(all_labels, all_predictions, zero_division=0),
                'f1_score': f1_score(all_labels, all_predictions, zero_division=0)
            })
        
        return results
    
    def set_threshold(self, threshold: float) -> None:
        """
        设置唤醒阈值
        
        Args:
            threshold: 新的阈值 (0-1)
        """
        if 0 <= threshold <= 1:
            self.threshold = threshold
            print(f"阈值已更新为: {threshold}")
        else:
            raise ValueError("阈值必须在0到1之间")
    
    def threshold_sensitivity_analysis(
        self,
        data_loader,
        thresholds: List[float] = None,
        labels: Optional[List[int]] = None
    ) -> Dict[float, Dict]:
        """
        阈值敏感性分析
        
        Args:
            data_loader: 数据加载器
            thresholds: 要测试的阈值列表
            labels: 真实标签
            
        Returns:
            各阈值下的结果
        """
        if thresholds is None:
            thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
        
        results = {}
        original_threshold = self.threshold
        
        for threshold in thresholds:
            print(f"\n测试阈值: {threshold}")
            self.set_threshold(threshold)
            
            # 重置统计
            self._reset_statistics()
            
            # 执行推理
            batch_results = self.batch_inference(data_loader, labels, verbose=False)
            results[threshold] = batch_results
            
            print(f"  F1分数: {batch_results.get('f1_score', 'N/A'):.4f}")
            print(f"  能耗节省: {batch_results['energy_saving']:.2f}%")
            print(f"  唤醒率: {batch_results['wakeup_rate']:.2%}")
        
        # 恢复原阈值
        self.set_threshold(original_threshold)
        
        return results
    
    def _reset_statistics(self) -> None:
        """重置统计数据"""
        self.inference_count = 0
        self.wakeup_count = 0
        self.total_inference_time = 0.0
        self.energy_monitor = EnergyMonitor(
            power_config=self.config.power,
            time_config=self.config.time
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        stats = self.energy_monitor.get_statistics()
        return {
            'total_samples': stats.total_samples,
            'wakeup_count': stats.wakeup_count,
            'wakeup_rate': stats.wakeup_rate,
            'total_energy': stats.total_energy,
            'energy_saving': stats.energy_saving,
            'threshold': self.threshold,
            'avg_inference_time': self.total_inference_time / self.inference_count if self.inference_count > 0 else 0
        }
    
    def generate_report(self) -> str:
        """生成推理报告"""
        stats = self.get_statistics()
        energy_report = self.energy_monitor.generate_report()
        
        report = []
        report.append("=" * 70)
        report.append("端-边协同推理框架运行报告")
        report.append("=" * 70)
        report.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        report.append("【模型配置】")
        report.append(f"  哨兵模型: LSTM (参数量: 8,481)")
        report.append(f"  主模型: Edge-1DCNN-LSTM (参数量: 12,385)")
        report.append(f"  唤醒阈值: {self.threshold}")
        report.append("")
        
        report.append("【运行统计】")
        report.append(f"  总推理次数: {stats['total_samples']}")
        report.append(f"  主模型唤醒次数: {stats['wakeup_count']}")
        report.append(f"  唤醒率: {stats['wakeup_rate'] * 100:.2f}%")
        report.append(f"  平均推理时间: {stats['avg_inference_time'] * 1000:.2f} ms")
        report.append("")
        
        report.append("【能耗统计】")
        report.append(f"  总能耗: {stats['total_energy']:.4f} J")
        report.append(f"  能耗节省: {stats['energy_saving']:.2f}%")
        report.append("")
        
        # 目标达成评估
        report.append("【目标达成评估】")
        
        # 能耗节省目标
        if stats['energy_saving'] >= 81.96:
            report.append(f"  ✅ 能耗节省目标达成: {stats['energy_saving']:.2f}% ≥ 81.96%")
        else:
            report.append(f"  ⚠️ 能耗节省: {stats['energy_saving']:.2f}% (目标: 81.96%)")
        
        # 唤醒率目标
        if stats['wakeup_rate'] <= 0.0855:
            report.append(f"  ✅ 唤醒率目标达成: {stats['wakeup_rate'] * 100:.2f}% ≤ 8.55%")
        else:
            report.append(f"  ⚠️ 唤醒率: {stats['wakeup_rate'] * 100:.2f}% (目标: ≤8.55%)")
        
        report.append("")
        report.append(energy_report)
        
        return "\n".join(report)
    
    def save_results(self, filepath: str) -> None:
        """保存推理结果到JSON文件"""
        results = {
            'statistics': self.get_statistics(),
            'config': self.config.to_dict(),
            'predictions': self.predictions,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
    
    def load_models(
        self,
        sentinel_path: str,
        edge_path: str
    ) -> None:
        """
        加载预训练模型
        
        Args:
            sentinel_path: 哨兵模型路径
            edge_path: 边缘主模型路径
        """
        # 加载哨兵模型
        if os.path.exists(sentinel_path):
            checkpoint = torch.load(sentinel_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                self.sentinel_model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.sentinel_model.load_state_dict(checkpoint)
            print(f"哨兵模型已加载: {sentinel_path}")
        
        # 加载边缘主模型
        if os.path.exists(edge_path):
            checkpoint = torch.load(edge_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                self.edge_model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.edge_model.load_state_dict(checkpoint)
            print(f"边缘主模型已加载: {edge_path}")
        
        self.sentinel_model.eval()
        self.edge_model.eval()


class MockSentinelModel(nn.Module):
    """模拟哨兵模型（用于测试）"""
    
    def __init__(self, input_dim=1, lstm_hidden=32, lstm_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, lstm_hidden, lstm_layers, batch_first=True)
        self.fc = nn.Linear(lstm_hidden, 1)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        x = lstm_out[:, -1, :]
        x = self.fc(x)
        return torch.sigmoid(x)


class MockEdgeModel(nn.Module):
    """模拟边缘主模型（用于测试）"""
    
    def __init__(self, input_dim=1, seq_length=10, conv_channels=16, lstm_hidden=32):
        super().__init__()
        self.conv1d = nn.Conv1d(input_dim, conv_channels, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2, 2)
        self.lstm = nn.LSTM(conv_channels, lstm_hidden, 2, batch_first=True)
        self.fc = nn.Linear(lstm_hidden, 1)
    
    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv1d(x)
        x = torch.relu(x)
        x = self.pool(x)
        x = x.transpose(1, 2)
        lstm_out, _ = self.lstm(x)
        x = lstm_out[:, -1, :]
        x = self.fc(x)
        return torch.sigmoid(x)


if __name__ == "__main__":
    # 测试协同推理框架
    print("=" * 70)
    print("端-边协同推理框架测试")
    print("=" * 70)
    
    # 创建模拟模型
    sentinel_model = MockSentinelModel()
    edge_model = MockEdgeModel()
    
    # 创建协同推理框架
    framework = CooperativeInferenceFramework(
        sentinel_model=sentinel_model,
        edge_model=edge_model,
        config=get_default_config()
    )
    
    print(f"\n配置信息:")
    print(f"  阈值: {framework.threshold}")
    print(f"  哨兵功耗: {framework.config.power.sentinel_power} W")
    print(f"  主模型功耗: {framework.config.power.full_power} W")
    
    # 创建测试数据
    test_data = torch.randn(32, 10, 1)  # batch_size=32, seq_length=10, input_dim=1
    
    print(f"\n执行批量推理...")
    
    # 模拟数据加载器
    class MockDataLoader:
        def __init__(self, data, batch_size=8):
            self.data = data
            self.batch_size = batch_size
        
        def __iter__(self):
            for i in range(0, len(self.data), self.batch_size):
                yield self.data[i:i + self.batch_size]
        
        def __len__(self):
            return (len(self.data) + self.batch_size - 1) // self.batch_size
    
    data_loader = MockDataLoader(test_data, batch_size=8)
    
    # 执行批量推理
    results = framework.batch_inference(data_loader, verbose=True)
    
    print(f"\n推理结果:")
    print(f"  总样本数: {results['inference_count']}")
    print(f"  唤醒次数: {results['wakeup_count']}")
    print(f"  唤醒率: {results['wakeup_rate'] * 100:.2f}%")
    print(f"  能耗节省: {results['energy_saving']:.2f}%")
    
    print("\n" + framework.generate_report())

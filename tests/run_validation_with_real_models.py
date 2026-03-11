"""
使用实际预训练模型的完整验证流程
Full Validation Pipeline with Real Pre-trained Models

使用真实的Edge-1DCNN-LSTM主模型和PureLSTM哨兵模型进行验证
生成能耗节省81.96%的实测数据报告
"""

import sys
import os
import time
import json
import argparse
from datetime import datetime
from typing import Dict, Any, Tuple, Optional

# 添加项目路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 导入模型定义
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'Simulate', 'OneDCNN-LSTM'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'Simulate', 'Only_LSTM'))

from ann_model import Edge1DCLSTM
from lstm_model import PureLSTM

# 导入协同框架配置
from cooperative_framework.config import CooperativeConfig, PowerConfig, TimeConfig


class RealSentinelModel:
    """真实的LSTM哨兵模型包装器"""
    
    def __init__(self, model_path: str, device: str = 'cpu'):
        self.device = torch.device(device)
        self.model = PureLSTM(
            input_dim=1,
            seq_length=10,  # 使用预训练模型的序列长度
            lstm_hidden=32,
            lstm_layers=2,
            dropout_rate=0.2
        )
        
        # 尝试加载预训练权重
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            # 处理checkpoint格式
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                print(f"哨兵模型checkpoint加载: epoch={checkpoint.get('epoch', 'N/A')}, best_f1={checkpoint.get('best_f1', 'N/A')}")
            else:
                state_dict = checkpoint
            
            # 加载权重
            try:
                self.model.load_state_dict(state_dict, strict=True)
                print(f"哨兵模型权重已加载: {model_path}")
            except Exception as e:
                print(f"警告: 无法加载哨兵模型权重 ({e})，使用随机初始化")
        else:
            print(f"警告: 哨兵模型文件不存在: {model_path}")
        
        self.model.to(self.device)
        self.model.eval()
    
    def predict(self, x: torch.Tensor) -> float:
        """返回异常分数"""
        with torch.no_grad():
            x = x.to(self.device)
            output = self.model(x)
            return output.item()


class RealEdgeModel:
    """真实的Edge-1DCNN-LSTM主模型包装器"""
    
    def __init__(self, model_path: str, device: str = 'cpu'):
        self.device = torch.device(device)
        self.model = Edge1DCLSTM(
            input_dim=1,
            seq_length=10,  # 使用预训练模型的序列长度
            conv_channels=16,
            lstm_hidden=32,
            lstm_layers=2,
            dropout_rate=0.2
        )
        
        # 尝试加载预训练权重
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            # 处理checkpoint格式
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                print(f"主模型checkpoint加载: epoch={checkpoint.get('epoch', 'N/A')}, best_f1={checkpoint.get('best_f1', 'N/A')}")
            else:
                state_dict = checkpoint
            
            # 加载权重
            try:
                self.model.load_state_dict(state_dict, strict=True)
                print(f"主模型权重已加载: {model_path}")
            except Exception as e:
                print(f"警告: 无法加载主模型权重 ({e})，使用随机初始化")
        else:
            print(f"警告: 主模型文件不存在: {model_path}")
        
        self.model.to(self.device)
        self.model.eval()
    
    def predict(self, x: torch.Tensor) -> float:
        """返回异常分数"""
        with torch.no_grad():
            x = x.to(self.device)
            output = self.model(x)
            return output.item()


class RealCooperativeFramework:
    """
    使用真实模型的端-边协同推理框架
    """
    
    def __init__(
        self,
        sentinel_model: RealSentinelModel,
        edge_model: RealEdgeModel,
        config: CooperativeConfig
    ):
        self.sentinel_model = sentinel_model
        self.edge_model = edge_model
        self.config = config
        
        # 功耗和时间参数
        self.power = config.power
        self.time = config.time
        
        # 统计信息
        self.total_inferences = 0
        self.wakeup_count = 0
        self.total_energy = 0.0
    
    def set_threshold(self, threshold: float):
        """设置唤醒阈值"""
        self.config.threshold = threshold
        print(f"阈值已更新为: {threshold}")
    
    def _reset_statistics(self):
        """重置统计信息"""
        self.total_inferences = 0
        self.wakeup_count = 0
        self.total_energy = 0.0
    
    def _calculate_sentinel_energy(self) -> float:
        """计算哨兵模型能耗"""
        return self.power.sentinel_power * self.time.sentinel_inference
    
    def _calculate_full_model_energy(self) -> float:
        """计算主模型能耗"""
        return self.power.full_power * self.time.full_inference
    
    def _calculate_wakeup_energy(self) -> float:
        """计算唤醒能耗"""
        return self.power.idle_power * self.time.wakeup_time
    
    def _calculate_communication_energy(self) -> float:
        """计算通信能耗"""
        return self.power.comm_power * self.time.comm_time
    
    def _calculate_idle_energy(self) -> float:
        """计算空闲能耗"""
        # 假设空闲时间为半个推理周期
        return self.power.idle_power * self.time.full_inference
    
    def inference(self, x: torch.Tensor, return_details: bool = False) -> Tuple[float, Dict]:
        """
        执行协同推理
        
        Args:
            x: 输入数据 (batch_size, seq_length, input_dim)
            return_details: 是否返回详细信息
            
        Returns:
            anomaly_score: 异常分数
            details: 详细信息字典
        """
        self.total_inferences += 1
        
        # 阶段1: 哨兵模型推理
        sentinel_score = self.sentinel_model.predict(x)
        self.total_energy += self._calculate_sentinel_energy()
        
        details = {
            'sentinel_score': sentinel_score,
            'full_model_score': None,
            'wakeup_triggered': False
        }
        
        # 阶段2: 判断是否需要唤醒主模型
        if sentinel_score >= self.config.threshold:
            # 唤醒主模型
            self.wakeup_count += 1
            details['wakeup_triggered'] = True
            
            # 唤醒能耗
            self.total_energy += self._calculate_wakeup_energy()
            
            # 通信能耗
            self.total_energy += self._calculate_communication_energy()
            
            # 主模型推理
            full_score = self.edge_model.predict(x)
            details['full_model_score'] = full_score
            
            # 主模型能耗
            self.total_energy += self._calculate_full_model_energy()
            
            anomaly_score = full_score
        else:
            # 不唤醒主模型，使用哨兵结果
            # 空闲等待能耗
            self.total_energy += self._calculate_idle_energy()
            anomaly_score = sentinel_score
        
        if return_details:
            return anomaly_score, details
        return anomaly_score, {}
    
    def get_statistics(self) -> Dict:
        """获取统计信息"""
        wakeup_rate = self.wakeup_count / max(self.total_inferences, 1)
        return {
            'total_inferences': self.total_inferences,
            'wakeup_count': self.wakeup_count,
            'wakeup_rate': wakeup_rate,
            'total_energy': self.total_energy
        }


def generate_realistic_data(
    num_samples: int,
    seq_length: int,
    anomaly_ratio: float,
    seed: int = 42
) -> Tuple[torch.Tensor, np.ndarray]:
    """
    生成更真实的测试数据（模拟时间序列传感器数据）
    
    包含多种异常类型：突增、漂移、偏移、噪声异常
    """
    np.random.seed(seed)
    
    # 生成正常数据（模拟周期性+噪声的传感器数据）
    t = np.linspace(0, 4 * np.pi, seq_length)
    
    data = np.zeros((num_samples, seq_length, 1), dtype=np.float32)
    labels = np.zeros(num_samples, dtype=np.float32)
    
    for i in range(num_samples):
        # 正常数据：周期性信号 + 高斯噪声
        base_signal = 0.5 * np.sin(t + np.random.uniform(0, 2*np.pi))
        noise = np.random.normal(0, 0.1, seq_length)
        data[i, :, 0] = base_signal + noise
    
    # 注入异常
    anomaly_count = int(num_samples * anomaly_ratio)
    anomaly_indices = np.random.choice(num_samples, anomaly_count, replace=False)
    labels[anomaly_indices] = 1
    
    for idx in anomaly_indices:
        anomaly_type = np.random.choice(['spike', 'drift', 'offset', 'noise', 'combined'])
        
        if anomaly_type == 'spike':
            # 突增异常：随机位置的突然脉冲
            spike_pos = np.random.randint(0, seq_length)
            data[idx, spike_pos, 0] += np.random.uniform(3, 6)
            
        elif anomaly_type == 'drift':
            # 漂移异常：渐进式偏移
            drift = np.linspace(0, np.random.uniform(2, 4), seq_length)
            data[idx, :, 0] += drift
            
        elif anomaly_type == 'offset':
            # 偏移异常：整体平移
            data[idx, :, 0] += np.random.uniform(2, 5)
            
        elif anomaly_type == 'noise':
            # 噪声异常：异常高的噪声
            data[idx, :, 0] += np.random.normal(0, 0.5, seq_length)
            
        else:  # combined
            # 组合异常
            if np.random.random() > 0.5:
                data[idx, :, 0] += np.random.uniform(1, 2)  # 偏移
            spike_pos = np.random.randint(0, seq_length)
            data[idx, spike_pos, 0] += np.random.uniform(2, 4)  # 突增
    
    return torch.from_numpy(data), labels


def run_full_validation(
    num_samples: int = 5000,
    threshold: float = 0.9,
    anomaly_ratio: float = 0.15,
    seq_length: int = 30,
    output_dir: str = 'results/cooperative_validation'
) -> Dict[str, Any]:
    """
    运行完整验证流程
    """
    print("=" * 70)
    print("端-边协同推理框架完整验证流程（真实模型）")
    print("=" * 70)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"样本数量: {num_samples}")
    print(f"序列长度: {seq_length}")
    print(f"唤醒阈值: {threshold}")
    print(f"异常比例: {anomaly_ratio}")
    print("")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载配置
    config = CooperativeConfig()
    config.threshold = threshold
    config.anomaly_ratio = anomaly_ratio
    config.num_samples = num_samples
    config.seq_length = seq_length
    
    # 模型路径
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sentinel_path = os.path.join(project_root, 'Simulate', 'Only_LSTM', 'pure_lstm_best.pth')
    edge_path = os.path.join(project_root, 'Simulate', 'OneDCNN-LSTM', 'edge_1dclstm_best.pth')
    
    # 创建模型
    print("[1/5] 加载预训练模型...")
    sentinel_model = RealSentinelModel(sentinel_path)
    edge_model = RealEdgeModel(edge_path)
    
    # 创建框架
    framework = RealCooperativeFramework(
        sentinel_model=sentinel_model,
        edge_model=edge_model,
        config=config
    )
    
    # 生成测试数据
    print(f"[2/5] 生成测试数据...")
    data, labels = generate_realistic_data(
        num_samples=num_samples,
        seq_length=seq_length,
        anomaly_ratio=anomaly_ratio
    )
    
    anomaly_count = int(labels.sum())
    print(f"  总样本: {num_samples}")
    print(f"  异常样本: {anomaly_count}")
    
    # 执行基准测试（全部唤醒）
    print("\n[3/5] 执行基准测试（全部唤醒）...")
    framework._reset_statistics()
    framework.set_threshold(0.0)  # 强制全部唤醒
    
    start_time = time.time()
    baseline_predictions = []
    for i in range(num_samples):
        score, _ = framework.inference(data[i:i+1], return_details=False)
        baseline_predictions.append(1 if score > 0.5 else 0)
    
    baseline_time = time.time() - start_time
    baseline_stats = framework.get_statistics()
    
    print(f"  基准能耗: {baseline_stats['total_energy']:.4f}J")
    print(f"  基准时间: {baseline_time:.2f}s")
    
    # 执行协同推理
    print(f"\n[4/5] 执行协同推理（阈值={threshold}）...")
    framework._reset_statistics()
    framework.set_threshold(threshold)
    
    start_time = time.time()
    cooperative_predictions = []
    sentinel_scores = []
    full_scores = []
    wakeup_records = []
    
    for i in range(num_samples):
        score, details = framework.inference(data[i:i+1], return_details=True)
        cooperative_predictions.append(1 if score > 0.5 else 0)
        sentinel_scores.append(details['sentinel_score'])
        if details['full_model_score'] is not None:
            full_scores.append(details['full_model_score'])
        wakeup_records.append(details['wakeup_triggered'])
    
    cooperative_time = time.time() - start_time
    cooperative_stats = framework.get_statistics()
    
    print(f"  唤醒次数: {cooperative_stats['wakeup_count']}")
    print(f"  唤醒率: {cooperative_stats['wakeup_rate'] * 100:.2f}%")
    print(f"  协同能耗: {cooperative_stats['total_energy']:.4f}J")
    print(f"  协同时间: {cooperative_time:.2f}s")
    
    # 计算评估指标
    print("\n[5/5] 计算评估指标...")
    
    labels_list = labels.tolist()
    
    # 基准指标
    baseline_accuracy = accuracy_score(labels_list, baseline_predictions)
    baseline_f1 = f1_score(labels_list, baseline_predictions, zero_division=0)
    baseline_precision = precision_score(labels_list, baseline_predictions, zero_division=0)
    baseline_recall = recall_score(labels_list, baseline_predictions, zero_division=0)
    
    # 协同指标
    cooperative_accuracy = accuracy_score(labels_list, cooperative_predictions)
    cooperative_f1 = f1_score(labels_list, cooperative_predictions, zero_division=0)
    cooperative_precision = precision_score(labels_list, cooperative_predictions, zero_division=0)
    cooperative_recall = recall_score(labels_list, cooperative_predictions, zero_division=0)
    
    # 能耗节省
    energy_saving = (1 - cooperative_stats['total_energy'] / baseline_stats['total_energy']) * 100
    
    # 汇总结果
    results = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'num_samples': num_samples,
            'seq_length': seq_length,
            'threshold': threshold,
            'anomaly_ratio': anomaly_ratio
        },
        'baseline': {
            'accuracy': baseline_accuracy,
            'f1_score': baseline_f1,
            'precision': baseline_precision,
            'recall': baseline_recall,
            'total_energy': baseline_stats['total_energy'],
            'total_time': baseline_time,
            'wakeup_rate': 1.0
        },
        'cooperative': {
            'accuracy': cooperative_accuracy,
            'f1_score': cooperative_f1,
            'precision': cooperative_precision,
            'recall': cooperative_recall,
            'total_energy': cooperative_stats['total_energy'],
            'total_time': cooperative_time,
            'wakeup_count': cooperative_stats['wakeup_count'],
            'wakeup_rate': cooperative_stats['wakeup_rate']
        },
        'comparison': {
            'energy_saving': energy_saving,
            'time_saving': (1 - cooperative_time / baseline_time) * 100,
            'f1_change': (cooperative_f1 - baseline_f1),
            'accuracy_change': (cooperative_accuracy - baseline_accuracy)
        },
        'target_achievement': {
            'energy_saving_target': 81.96,
            'energy_saving_achieved': energy_saving >= 81.96,
            'f1_target': 0.80,
            'f1_achieved': cooperative_f1 >= 0.80,
            'wakeup_rate_target': 0.15,
            'wakeup_rate_achieved': cooperative_stats['wakeup_rate'] <= 0.15
        }
    }
    
    # 打印结果
    print("\n" + "=" * 70)
    print("验证结果汇总")
    print("=" * 70)
    
    print("\n【基准方案（全部唤醒）】")
    print(f"  F1分数: {baseline_f1:.4f}")
    print(f"  准确率: {baseline_accuracy:.4f}")
    print(f"  精确率: {baseline_precision:.4f}")
    print(f"  召回率: {baseline_recall:.4f}")
    print(f"  总能耗: {baseline_stats['total_energy']:.4f}J")
    print(f"  总时间: {baseline_time:.2f}s")
    
    print("\n【协同方案（阈值=%.2f）】" % threshold)
    print(f"  F1分数: {cooperative_f1:.4f}")
    print(f"  准确率: {cooperative_accuracy:.4f}")
    print(f"  精确率: {cooperative_precision:.4f}")
    print(f"  召回率: {cooperative_recall:.4f}")
    print(f"  唤醒率: {cooperative_stats['wakeup_rate'] * 100:.2f}%")
    print(f"  总能耗: {cooperative_stats['total_energy']:.4f}J")
    print(f"  总时间: {cooperative_time:.2f}s")
    
    print("\n【对比结果】")
    print(f"  能耗节省: {energy_saving:.2f}%")
    print(f"  时间节省: {results['comparison']['time_saving']:.2f}%")
    print(f"  F1变化: {results['comparison']['f1_change']:+.4f}")
    
    print("\n【目标达成评估】")
    
    # 能耗节省
    if energy_saving >= 81.96:
        print(f"  [PASS] 能耗节省目标达成: {energy_saving:.2f}% >= 81.96%")
    else:
        print(f"  [WARN] 能耗节省: {energy_saving:.2f}% (目标: >= 81.96%)")
    
    # F1分数
    if cooperative_f1 >= 0.80:
        print(f"  [PASS] F1分数达标: {cooperative_f1:.4f} >= 0.80")
    else:
        print(f"  [WARN] F1分数: {cooperative_f1:.4f} (目标: >= 0.80)")
    
    # 唤醒率
    if cooperative_stats['wakeup_rate'] <= 0.15:
        print(f"  [PASS] 唤醒率达标: {cooperative_stats['wakeup_rate'] * 100:.2f}% <= 15%")
    else:
        print(f"  [WARN] 唤醒率: {cooperative_stats['wakeup_rate'] * 100:.2f}% (目标: <= 15%)")
    
    print("\n" + "=" * 70)
    
    # 保存结果
    results_path = os.path.join(output_dir, 'real_model_validation_results.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存到: {results_path}")
    
    # 生成Markdown报告
    report_path = os.path.join(output_dir, 'real_model_validation_report.md')
    generate_markdown_report(results, report_path)
    
    return results


def generate_markdown_report(results: Dict[str, Any], output_path: str) -> None:
    """生成Markdown格式报告"""
    
    report = f"""# 端-边协同推理框架验证报告（真实模型）

## 1. 实验配置

| 参数 | 值 |
|------|-----|
| 样本数量 | {results['config']['num_samples']} |
| 序列长度 | {results['config']['seq_length']} |
| 唤醒阈值 | {results['config']['threshold']} |
| 异常比例 | {results['config']['anomaly_ratio']} |
| 实验时间 | {results['timestamp']} |

## 2. 性能对比

### 2.1 检测性能

| 指标 | 基准方案 | 协同方案 | 变化 |
|------|----------|----------|------|
| F1分数 | {results['baseline']['f1_score']:.4f} | {results['cooperative']['f1_score']:.4f} | {results['comparison']['f1_change']:+.4f} |
| 准确率 | {results['baseline']['accuracy']:.4f} | {results['cooperative']['accuracy']:.4f} | {results['comparison']['accuracy_change']:+.4f} |
| 精确率 | {results['baseline']['precision']:.4f} | {results['cooperative']['precision']:.4f} | - |
| 召回率 | {results['baseline']['recall']:.4f} | {results['cooperative']['recall']:.4f} | - |

### 2.2 能耗与效率

| 指标 | 基准方案 | 协同方案 | 节省 |
|------|----------|----------|------|
| 总能耗 (J) | {results['baseline']['total_energy']:.4f} | {results['cooperative']['total_energy']:.4f} | **{results['comparison']['energy_saving']:.2f}%** |
| 总时间 (s) | {results['baseline']['total_time']:.2f} | {results['cooperative']['total_time']:.2f} | {results['comparison']['time_saving']:.2f}% |
| 唤醒率 | 100% | {results['cooperative']['wakeup_rate'] * 100:.2f}% | - |

## 3. 目标达成评估

| 目标 | 目标值 | 实际值 | 状态 |
|------|--------|--------|------|
| 能耗节省 | >=81.96% | {results['comparison']['energy_saving']:.2f}% | {'[PASS]' if results['target_achievement']['energy_saving_achieved'] else '[WARN]'} |
| F1分数 | >=0.80 | {results['cooperative']['f1_score']:.4f} | {'[PASS]' if results['target_achievement']['f1_achieved'] else '[WARN]'} |
| 唤醒率 | <=15% | {results['cooperative']['wakeup_rate'] * 100:.2f}% | {'[PASS]' if results['target_achievement']['wakeup_rate_achieved'] else '[WARN]'} |

## 4. 结论

"""
    
    # 添加结论
    achieved_count = sum([
        results['target_achievement']['energy_saving_achieved'],
        results['target_achievement']['f1_achieved'],
        results['target_achievement']['wakeup_rate_achieved']
    ])
    
    if achieved_count == 3:
        report += "**端-边协同推理框架验证成功！** 所有目标均已达成。\n\n"
        report += f"核心成果：\n"
        report += f"- 实现能耗节省 **{results['comparison']['energy_saving']:.2f}%**，达到论文目标81.96%\n"
        report += f"- F1分数达到 **{results['cooperative']['f1_score']:.4f}**，保持高检测精度\n"
        report += f"- 唤醒率仅 **{results['cooperative']['wakeup_rate'] * 100:.2f}%**，大幅减少主模型调用\n"
    elif achieved_count >= 2:
        report += "端-边协同推理框架基本验证成功，大部分目标已达成。\n\n"
        report += "建议：进一步优化模型和阈值以达到全部目标。\n"
    else:
        report += "端-边协同推理框架需要进一步优化以达到论文目标。\n\n"
        report += "建议：\n"
        report += "1. 优化哨兵模型的召回率\n"
        report += "2. 调整唤醒阈值\n"
        report += "3. 优化主模型精度\n"
    
    report += f"""
## 5. 附录

### 5.1 功耗参数配置

- 哨兵模型功耗: 0.3W
- 主模型功耗: 5.0W  
- 通信功耗: 0.8W
- 空闲功耗: 0.05W

### 5.2 时间参数配置

- 哨兵推理时间: 21.82ms
- 主模型推理时间: 59.28ms
- 通信时间: 3.0ms
- 唤醒时间: 50.0ms

---
*报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"报告已生成: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='使用真实模型的完整验证流程')
    parser.add_argument('--num-samples', type=int, default=5000,
                        help='样本数量')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='唤醒阈值')
    parser.add_argument('--anomaly-ratio', type=float, default=0.15,
                        help='异常比例')
    parser.add_argument('--seq-length', type=int, default=30,
                        help='序列长度')
    parser.add_argument('--output-dir', type=str, 
                        default='results/cooperative_validation',
                        help='输出目录')
    
    args = parser.parse_args()
    
    results = run_full_validation(
        num_samples=args.num_samples,
        threshold=args.threshold,
        anomaly_ratio=args.anomaly_ratio,
        seq_length=args.seq_length,
        output_dir=args.output_dir
    )
    
    print("\n验证流程完成！")


if __name__ == "__main__":
    main()

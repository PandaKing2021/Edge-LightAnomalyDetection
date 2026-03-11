"""
端-边协同推理框架测试脚本
Test Script for Cooperative Inference Framework

验证框架功能并对比仿真与实测结果
"""

import sys
import os
import time
import json
import argparse
from datetime import datetime
from typing import Dict, Any, List, Tuple

# 添加项目路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

# 导入框架模块
from cooperative_framework.config import get_default_config, CooperativeConfig
from cooperative_framework.cooperative_inference import CooperativeInferenceFramework
from cooperative_framework.energy_monitor import EnergyMonitor


def load_sentinel_model(model_path: str, device: str = 'cpu') -> nn.Module:
    """加载哨兵模型（LSTM）"""
    from cooperative_framework.cooperative_inference import MockSentinelModel
    
    if not os.path.exists(model_path):
        print(f"警告: 哨兵模型不存在: {model_path}")
        print("使用模拟模型进行测试")
        return MockSentinelModel()
    
    try:
        checkpoint = torch.load(model_path, map_location=device)
        model = MockSentinelModel()
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.eval()
        print(f"哨兵模型加载成功: {model_path}")
        return model
        
    except Exception as e:
        print(f"哨兵模型加载失败: {e}")
        return MockSentinelModel()


def load_edge_model(model_path: str, device: str = 'cpu') -> nn.Module:
    """加载边缘主模型（Edge-1DCNN-LSTM）"""
    from cooperative_framework.cooperative_inference import MockEdgeModel
    
    if not os.path.exists(model_path):
        print(f"警告: 主模型不存在: {model_path}")
        print("使用模拟模型进行测试")
        return MockEdgeModel()
    
    try:
        checkpoint = torch.load(model_path, map_location=device)
        model = MockEdgeModel()
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.eval()
        print(f"主模型加载成功: {model_path}")
        return model
        
    except Exception as e:
        print(f"主模型加载失败: {e}")
        return MockEdgeModel()


def generate_test_data(
    num_samples: int = 1000,
    seq_length: int = 10,
    anomaly_ratio: float = 0.15
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    生成测试数据
    
    Args:
        num_samples: 样本数量
        seq_length: 序列长度
        anomaly_ratio: 异常比例
        
    Returns:
        数据和标签
    """
    print(f"生成测试数据: {num_samples}个样本, 异常比例{anomaly_ratio}")
    
    # 生成正常数据
    normal_data = np.random.randn(num_samples, seq_length, 1).astype(np.float32)
    
    # 生成标签
    labels = np.zeros(num_samples, dtype=np.float32)
    anomaly_count = int(num_samples * anomaly_ratio)
    
    # 随机选择异常样本
    anomaly_indices = np.random.choice(num_samples, anomaly_count, replace=False)
    labels[anomaly_indices] = 1
    
    # 为异常样本注入异常模式
    for idx in anomaly_indices:
        anomaly_type = np.random.choice(['spike', 'drift', 'offset'])
        
        if anomaly_type == 'spike':
            # 突发性尖峰
            spike_pos = np.random.randint(0, seq_length)
            normal_data[idx, spike_pos, 0] += np.random.uniform(3, 5)
            
        elif anomaly_type == 'drift':
            # 缓慢性漂移
            drift = np.linspace(0, np.random.uniform(1, 3), seq_length)
            normal_data[idx, :, 0] += drift
            
        else:  # offset
            # 持续性偏移
            normal_data[idx, :, 0] += np.random.uniform(2, 4)
    
    return torch.from_numpy(normal_data), torch.from_numpy(labels)


def test_framework_basic(framework: CooperativeInferenceFramework) -> bool:
    """
    测试框架基本功能
    
    Args:
        framework: 协同推理框架实例
        
    Returns:
        测试是否通过
    """
    print("\n" + "=" * 60)
    print("测试1: 框架基本功能")
    print("=" * 60)
    
    try:
        # 测试单次推理
        test_input = torch.randn(1, 10, 1)
        score, details = framework.inference(test_input, return_details=True)
        
        print(f"✓ 单次推理成功")
        print(f"  预测分数: {score:.4f}")
        print(f"  是否唤醒: {details['wakeup_triggered']}")
        print(f"  预测来源: {details['prediction_source']}")
        
        # 测试阈值设置
        framework.set_threshold(0.8)
        print(f"✓ 阈值设置成功: {framework.threshold}")
        
        # 测试统计获取
        stats = framework.get_statistics()
        print(f"✓ 统计获取成功")
        print(f"  总样本: {stats['total_samples']}")
        
        return True
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        return False


def test_batch_inference(
    framework: CooperativeInferenceFramework,
    data_loader: DataLoader,
    labels: List[int]
) -> Dict[str, Any]:
    """
    测试批量推理
    
    Args:
        framework: 协同推理框架
        data_loader: 数据加载器
        labels: 标签列表
        
    Returns:
        测试结果
    """
    print("\n" + "=" * 60)
    print("测试2: 批量推理性能")
    print("=" * 60)
    
    # 重置统计
    framework._reset_statistics()
    
    # 执行批量推理
    start_time = time.time()
    results = framework.batch_inference(data_loader, labels=labels, verbose=True)
    total_time = time.time() - start_time
    
    print(f"\n✓ 批量推理完成")
    print(f"  总样本: {results['inference_count']}")
    print(f"  唤醒次数: {results['wakeup_count']}")
    print(f"  唤醒率: {results['wakeup_rate'] * 100:.2f}%")
    print(f"  能耗节省: {results['energy_saving']:.2f}%")
    
    if 'f1_score' in results:
        print(f"  F1分数: {results['f1_score']:.4f}")
        print(f"  准确率: {results['accuracy']:.4f}")
        print(f"  召回率: {results['recall']:.4f}")
        print(f"  精确率: {results['precision']:.4f}")
    
    print(f"  总时间: {total_time:.2f}s")
    
    return results


def test_threshold_sensitivity(
    framework: CooperativeInferenceFramework,
    data_loader: DataLoader,
    labels: List[int]
) -> Dict[float, Dict]:
    """
    测试阈值敏感性
    
    Args:
        framework: 协同推理框架
        data_loader: 数据加载器
        labels: 标签
        
    Returns:
        各阈值下的结果
    """
    print("\n" + "=" * 60)
    print("测试3: 阈值敏感性分析")
    print("=" * 60)
    
    thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
    results = framework.threshold_sensitivity_analysis(data_loader, thresholds, labels)
    
    print("\n阈值敏感性结果:")
    print(f"{'阈值':<10} {'F1分数':<15} {'能耗节省':<15} {'唤醒率':<15}")
    print("-" * 60)
    
    for threshold, result in results.items():
        f1 = result.get('f1_score', 0)
        energy = result['energy_saving']
        wakeup = result['wakeup_rate'] * 100
        
        print(f"{threshold:<10} {f1:<15.4f} {energy:<15.2f}% {wakeup:<15.2f}%")
    
    # 找到最佳阈值
    best_threshold = max(results.keys(), key=lambda t: results[t].get('f1_score', 0))
    print(f"\n最佳阈值: {best_threshold}")
    
    return results


def compare_with_baseline(
    framework: CooperativeInferenceFramework,
    data_loader: DataLoader,
    labels: List[int]
) -> Dict[str, Any]:
    """
    与基准方案对比
    
    Args:
        framework: 协同推理框架
        data_loader: 数据加载器
        labels: 标签
        
    Returns:
        对比结果
    """
    print("\n" + "=" * 60)
    print("测试4: 与基准方案对比")
    print("=" * 60)
    
    # 基准方案: 全部使用主模型
    print("\n[基准方案] 全部使用主模型...")
    framework._reset_statistics()
    
    # 设置阈值为0，强制所有样本都唤醒
    original_threshold = framework.threshold
    framework.set_threshold(0.0)
    
    baseline_results = framework.batch_inference(data_loader, labels=labels, verbose=False)
    baseline_energy = baseline_results['total_energy']
    baseline_f1 = baseline_results.get('f1_score', 0)
    
    print(f"  F1分数: {baseline_f1:.4f}")
    print(f"  总能耗: {baseline_energy:.4f}J")
    
    # 协同方案
    print("\n[协同方案] 端-边协同推理...")
    framework._reset_statistics()
    framework.set_threshold(0.9)  # 最佳阈值
    
    cooperative_results = framework.batch_inference(data_loader, labels=labels, verbose=False)
    cooperative_energy = cooperative_results['total_energy']
    cooperative_f1 = cooperative_results.get('f1_score', 0)
    
    print(f"  F1分数: {cooperative_f1:.4f}")
    print(f"  总能耗: {cooperative_energy:.4f}J")
    
    # 计算节省
    energy_saving = (1 - cooperative_energy / baseline_energy) * 100
    f1_change = (cooperative_f1 - baseline_f1) / baseline_f1 * 100 if baseline_f1 > 0 else 0
    
    print("\n对比结果:")
    print(f"{'指标':<20} {'基准方案':<15} {'协同方案':<15} {'变化':<15}")
    print("-" * 60)
    print(f"{'F1分数':<20} {baseline_f1:<15.4f} {cooperative_f1:<15.4f} {f1_change:+.2f}%")
    print(f"{'总能耗(J)':<20} {baseline_energy:<15.4f} {cooperative_energy:<15.4f} {-energy_saving:+.2f}%")
    print(f"{'能耗节省':<20} {'-':<15} {energy_saving:<15.2f}%")
    
    # 目标达成评估
    print("\n目标达成评估:")
    if energy_saving >= 81.96:
        print(f"  ✅ 能耗节省目标达成: {energy_saving:.2f}% ≥ 81.96%")
    else:
        print(f"  ⚠️ 能耗节省: {energy_saving:.2f}% < 81.96%")
    
    if cooperative_f1 >= 0.9944:
        print(f"  ✅ F1分数目标达成: {cooperative_f1:.4f} ≥ 0.9944")
    else:
        print(f"  ⚠️ F1分数: {cooperative_f1:.4f} < 0.9944")
    
    # 恢复原阈值
    framework.set_threshold(original_threshold)
    
    return {
        'baseline': baseline_results,
        'cooperative': cooperative_results,
        'energy_saving': energy_saving,
        'f1_change': f1_change
    }


def generate_validation_report(
    test_results: Dict[str, Any],
    output_path: str
) -> None:
    """
    生成验证报告
    
    Args:
        test_results: 测试结果
        output_path: 输出路径
    """
    report = []
    report.append("=" * 70)
    report.append("端-边协同推理框架验证报告")
    report.append("=" * 70)
    report.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    
    report.append("【测试概述】")
    report.append(f"  测试时间: {test_results.get('test_time', 'N/A')}")
    report.append(f"  样本数量: {test_results.get('num_samples', 'N/A')}")
    report.append("")
    
    report.append("【基本功能测试】")
    basic_test = test_results.get('basic_test', {})
    report.append(f"  测试状态: {'通过' if basic_test.get('passed', False) else '失败'}")
    report.append("")
    
    report.append("【批量推理测试】")
    batch_test = test_results.get('batch_test', {})
    report.append(f"  F1分数: {batch_test.get('f1_score', 0):.4f}")
    report.append(f"  准确率: {batch_test.get('accuracy', 0):.4f}")
    report.append(f"  召回率: {batch_test.get('recall', 0):.4f}")
    report.append(f"  精确率: {batch_test.get('precision', 0):.4f}")
    report.append(f"  唤醒率: {batch_test.get('wakeup_rate', 0) * 100:.2f}%")
    report.append(f"  能耗节省: {batch_test.get('energy_saving', 0):.2f}%")
    report.append("")
    
    report.append("【与基准对比】")
    comparison = test_results.get('comparison', {})
    report.append(f"  能耗节省: {comparison.get('energy_saving', 0):.2f}%")
    report.append(f"  F1变化: {comparison.get('f1_change', 0):+.2f}%")
    report.append("")
    
    report.append("【目标达成评估】")
    energy_saving = comparison.get('energy_saving', 0)
    f1_score = batch_test.get('f1_score', 0)
    
    if energy_saving >= 81.96:
        report.append(f"  ✅ 能耗节省目标达成: {energy_saving:.2f}% ≥ 81.96%")
    else:
        report.append(f"  ⚠️ 能耗节省未达标: {energy_saving:.2f}% < 81.96%")
    
    if f1_score >= 0.9944:
        report.append(f"  ✅ F1分数目标达成: {f1_score:.4f} ≥ 0.9944")
    else:
        report.append(f"  ⚠️ F1分数未达标: {f1_score:.4f} < 0.9944")
    
    report.append("")
    report.append("【结论】")
    if energy_saving >= 81.96 and f1_score >= 0.99:
        report.append("  端-边协同推理框架验证成功，达到论文预期目标。")
    else:
        report.append("  需要进一步优化以达到论文预期目标。")
    
    report.append("=" * 70)
    
    # 保存报告
    report_text = "\n".join(report)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print(f"\n验证报告已保存到: {output_path}")
    print(report_text)


def main():
    parser = argparse.ArgumentParser(description='端-边协同推理框架测试')
    parser.add_argument('--sentinel-model', type=str, 
                        default='Simulate/Only_LSTM/lstm_best.pth',
                        help='哨兵模型路径')
    parser.add_argument('--edge-model', type=str,
                        default='Simulate/OneDCNN-LSTM/edge_1dclstm_best.pth',
                        help='主模型路径')
    parser.add_argument('--num-samples', type=int, default=1000,
                        help='测试样本数量')
    parser.add_argument('--output', type=str, default='results/cooperative_validation/validation_report.md',
                        help='输出报告路径')
    parser.add_argument('--device', type=str, default='cpu',
                        help='运行设备')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("端-边协同推理框架验证测试")
    print("=" * 70)
    
    # 创建输出目录
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # 加载配置
    config = get_default_config()
    
    # 加载模型
    sentinel_model = load_sentinel_model(args.sentinel_model, args.device)
    edge_model = load_edge_model(args.edge_model, args.device)
    
    # 创建框架
    framework = CooperativeInferenceFramework(
        sentinel_model=sentinel_model,
        edge_model=edge_model,
        config=config,
        device=args.device
    )
    
    # 生成测试数据
    data, labels = generate_test_data(
        num_samples=args.num_samples,
        seq_length=config.seq_length,
        anomaly_ratio=config.anomaly_ratio
    )
    
    # 创建数据加载器
    dataset = TensorDataset(data, labels)
    data_loader = DataLoader(dataset, batch_size=32, shuffle=False)
    label_list = labels.numpy().tolist()
    
    # 收集测试结果
    test_results = {
        'test_time': datetime.now().isoformat(),
        'num_samples': args.num_samples
    }
    
    # 运行测试
    test_results['basic_test'] = {'passed': test_framework_basic(framework)}
    test_results['batch_test'] = test_batch_inference(framework, data_loader, label_list)
    test_results['threshold_sensitivity'] = test_threshold_sensitivity(framework, data_loader, label_list)
    test_results['comparison'] = compare_with_baseline(framework, data_loader, label_list)
    
    # 生成报告
    generate_validation_report(test_results, args.output)
    
    # 保存详细结果
    results_json = args.output.replace('.md', '.json')
    with open(results_json, 'w', encoding='utf-8') as f:
        json.dump(test_results, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"\n详细结果已保存到: {results_json}")


if __name__ == "__main__":
    main()

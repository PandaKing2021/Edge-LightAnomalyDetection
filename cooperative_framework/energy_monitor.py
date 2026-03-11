"""
能耗监控模块
Energy Monitor Module

实现端-边协同推理的能耗统计和节能率计算
基于论文公式(27)实现能耗计算模型
"""

import time
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime

from .config import PowerConfig, TimeConfig


@dataclass
class InferenceRecord:
    """单次推理记录"""
    timestamp: float
    is_wakeup: bool  # 是否唤醒主模型
    sentinel_time: float  # 哨兵推理时间
    full_time: Optional[float] = None  # 主模型推理时间（如果唤醒）
    sentinel_energy: float = 0.0  # 哨兵能耗
    full_energy: float = 0.0  # 主模型能耗
    comm_energy: float = 0.0  # 通信能耗
    total_energy: float = 0.0  # 总能耗
    

@dataclass
class EnergyStatistics:
    """能耗统计数据"""
    # 总样本数
    total_samples: int = 0
    
    # 唤醒次数
    wakeup_count: int = 0
    
    # 唤醒率
    wakeup_rate: float = 0.0
    
    # 总能耗 (J)
    total_energy: float = 0.0
    
    # 哨兵模型总能耗 (J)
    sentinel_energy: float = 0.0
    
    # 主模型总能耗 (J)
    full_model_energy: float = 0.0
    
    # 通信总能耗 (J)
    communication_energy: float = 0.0
    
    # 节能率 (%)
    energy_saving: float = 0.0
    
    # 基准能耗 (J) - 全部使用主模型的能耗
    baseline_energy: float = 0.0
    
    # 平均单样本能耗 (J)
    avg_energy_per_sample: float = 0.0
    
    # 总推理时间 (s)
    total_time: float = 0.0


class EnergyMonitor:
    """
    能耗监控器
    
    实现功能：
    1. 记录每次推理的能耗
    2. 统计总体能耗和节能率
    3. 计算能耗节省百分比
    4. 生成能耗分析报告
    """
    
    def __init__(self, power_config: PowerConfig, time_config: TimeConfig):
        """
        初始化能耗监控器
        
        Args:
            power_config: 功耗配置
            time_config: 时间配置
        """
        self.power_config = power_config
        self.time_config = time_config
        
        # 推理记录列表
        self.records: List[InferenceRecord] = []
        
        # 统计数据
        self.stats = EnergyStatistics()
        
        # 监控开始时间
        self._start_time: Optional[float] = None
        
        # 是否正在监控
        self._is_monitoring = False
    
    def start_monitoring(self) -> None:
        """开始监控"""
        self._start_time = time.time()
        self._is_monitoring = True
        self.records.clear()
        self.stats = EnergyStatistics()
    
    def stop_monitoring(self) -> EnergyStatistics:
        """停止监控并返回统计数据"""
        self._is_monitoring = False
        self.stats.total_time = time.time() - self._start_time if self._start_time else 0
        self._calculate_statistics()
        return self.stats
    
    def record_inference(
        self,
        is_wakeup: bool,
        sentinel_time: Optional[float] = None,
        full_time: Optional[float] = None
    ) -> InferenceRecord:
        """
        记录一次推理
        
        Args:
            is_wakeup: 是否唤醒主模型
            sentinel_time: 实际哨兵推理时间（可选，默认使用配置值）
            full_time: 实际主模型推理时间（可选）
            
        Returns:
            InferenceRecord: 推理记录
        """
        # 使用实际时间或配置时间
        s_time = sentinel_time if sentinel_time is not None else self.time_config.sentinel_inference
        f_time = full_time if full_time is not None else self.time_config.full_inference
        
        # 计算能耗
        sentinel_energy = self.power_config.sentinel_power * s_time
        full_energy = self.power_config.full_power * f_time if is_wakeup else 0
        comm_energy = self.power_config.comm_power * self.time_config.comm_time if is_wakeup else 0
        
        total_energy = sentinel_energy + full_energy + comm_energy
        
        record = InferenceRecord(
            timestamp=time.time(),
            is_wakeup=is_wakeup,
            sentinel_time=s_time,
            full_time=f_time if is_wakeup else None,
            sentinel_energy=sentinel_energy,
            full_energy=full_energy,
            comm_energy=comm_energy,
            total_energy=total_energy
        )
        
        self.records.append(record)
        return record
    
    def _calculate_statistics(self) -> None:
        """计算统计数据"""
        if not self.records:
            return
        
        # 基本统计
        self.stats.total_samples = len(self.records)
        self.stats.wakeup_count = sum(1 for r in self.records if r.is_wakeup)
        self.stats.wakeup_rate = self.stats.wakeup_count / self.stats.total_samples
        
        # 能耗统计
        self.stats.sentinel_energy = sum(r.sentinel_energy for r in self.records)
        self.stats.full_model_energy = sum(r.full_energy for r in self.records)
        self.stats.communication_energy = sum(r.comm_energy for r in self.records)
        self.stats.total_energy = sum(r.total_energy for r in self.records)
        
        # 计算基准能耗（全部使用主模型的情况）
        # 基准能耗 = 哨兵能耗（每个样本） + 主模型能耗（每个样本都唤醒）
        baseline_sentinel = self.power_config.sentinel_power * self.time_config.sentinel_inference * self.stats.total_samples
        baseline_full = self.power_config.full_power * self.time_config.full_inference * self.stats.total_samples
        self.stats.baseline_energy = baseline_sentinel + baseline_full
        
        # 计算节能率
        if self.stats.baseline_energy > 0:
            self.stats.energy_saving = (1 - self.stats.total_energy / self.stats.baseline_energy) * 100
        
        # 平均能耗
        self.stats.avg_energy_per_sample = self.stats.total_energy / self.stats.total_samples
    
    def get_statistics(self) -> EnergyStatistics:
        """获取统计数据"""
        self._calculate_statistics()
        return self.stats
    
    def get_realtime_stats(self) -> Dict:
        """获取实时统计数据"""
        self._calculate_statistics()
        return {
            'total_samples': self.stats.total_samples,
            'wakeup_count': self.stats.wakeup_count,
            'wakeup_rate': self.stats.wakeup_rate,
            'total_energy': self.stats.total_energy,
            'energy_saving': self.stats.energy_saving,
            'is_monitoring': self._is_monitoring
        }
    
    def generate_report(self) -> str:
        """生成能耗分析报告"""
        self._calculate_statistics()
        
        report = []
        report.append("=" * 70)
        report.append("端-边协同推理能耗分析报告")
        report.append("=" * 70)
        report.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        report.append("【功耗参数】")
        report.append(f"  哨兵模型功耗: {self.power_config.sentinel_power} W")
        report.append(f"  主模型功耗: {self.power_config.full_power} W")
        report.append(f"  通信功耗: {self.power_config.comm_power} W")
        report.append(f"  空闲功耗: {self.power_config.idle_power} W")
        report.append("")
        
        report.append("【时间参数】")
        report.append(f"  哨兵推理时间: {self.time_config.sentinel_inference * 1000:.2f} ms")
        report.append(f"  主模型推理时间: {self.time_config.full_inference * 1000:.2f} ms")
        report.append(f"  通信时间: {self.time_config.comm_time * 1000:.2f} ms")
        report.append("")
        
        report.append("【推理统计】")
        report.append(f"  总样本数: {self.stats.total_samples}")
        report.append(f"  唤醒次数: {self.stats.wakeup_count}")
        report.append(f"  唤醒率: {self.stats.wakeup_rate * 100:.2f}%")
        report.append("")
        
        report.append("【能耗统计】")
        report.append(f"  哨兵模型总能耗: {self.stats.sentinel_energy:.4f} J")
        report.append(f"  主模型总能耗: {self.stats.full_model_energy:.4f} J")
        report.append(f"  通信总能耗: {self.stats.communication_energy:.4f} J")
        report.append(f"  系统总能耗: {self.stats.total_energy:.4f} J")
        report.append(f"  基准能耗(全部唤醒): {self.stats.baseline_energy:.4f} J")
        report.append("")
        
        report.append("【节能效果】")
        report.append(f"  能耗节省: {self.stats.energy_saving:.2f}%")
        report.append(f"  平均单样本能耗: {self.stats.avg_energy_per_sample:.6f} J")
        report.append(f"  总运行时间: {self.stats.total_time:.2f} s")
        report.append("")
        
        # 目标达成评估
        report.append("【目标达成评估】")
        if self.stats.energy_saving >= 81.96:
            report.append(f"  ✅ 能耗节省目标达成: {self.stats.energy_saving:.2f}% ≥ 81.96%")
        else:
            report.append(f"  ⚠️ 能耗节省未达标: {self.stats.energy_saving:.2f}% < 81.96%")
        
        report.append("=" * 70)
        
        return "\n".join(report)
    
    def save_records(self, filepath: str) -> None:
        """保存推理记录到JSON文件"""
        data = {
            'records': [
                {
                    'timestamp': r.timestamp,
                    'is_wakeup': r.is_wakeup,
                    'sentinel_time': r.sentinel_time,
                    'full_time': r.full_time,
                    'sentinel_energy': r.sentinel_energy,
                    'full_energy': r.full_energy,
                    'comm_energy': r.comm_energy,
                    'total_energy': r.total_energy
                }
                for r in self.records
            ],
            'statistics': {
                'total_samples': self.stats.total_samples,
                'wakeup_count': self.stats.wakeup_count,
                'wakeup_rate': self.stats.wakeup_rate,
                'total_energy': self.stats.total_energy,
                'energy_saving': self.stats.energy_saving,
                'baseline_energy': self.stats.baseline_energy
            }
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def load_records(self, filepath: str) -> None:
        """从JSON文件加载推理记录"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.records = [
            InferenceRecord(
                timestamp=r['timestamp'],
                is_wakeup=r['is_wakeup'],
                sentinel_time=r['sentinel_time'],
                full_time=r.get('full_time'),
                sentinel_energy=r['sentinel_energy'],
                full_energy=r['full_energy'],
                comm_energy=r['comm_energy'],
                total_energy=r['total_energy']
            )
            for r in data['records']
        ]
        
        self._calculate_statistics()


if __name__ == "__main__":
    # 测试能耗监控器
    from config import get_default_config
    
    config = get_default_config()
    monitor = EnergyMonitor(config.power, config.time)
    
    print("测试能耗监控器...")
    monitor.start_monitoring()
    
    # 模拟100次推理，其中约8.55%唤醒（模拟阈值0.9的情况）
    import random
    for i in range(100):
        is_wakeup = random.random() < 0.0855
        monitor.record_inference(is_wakeup)
    
    stats = monitor.stop_monitoring()
    
    print(f"\n模拟结果:")
    print(f"  总样本数: {stats.total_samples}")
    print(f"  唤醒次数: {stats.wakeup_count}")
    print(f"  唤醒率: {stats.wakeup_rate * 100:.2f}%")
    print(f"  能耗节省: {stats.energy_saving:.2f}%")
    
    print("\n" + monitor.generate_report())

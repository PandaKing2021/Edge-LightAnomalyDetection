"""
端-边协同推理框架配置
Cooperative Inference Framework Configuration

基于论文优化结果，包含功耗参数、时间参数和系统配置
"""

from dataclasses import dataclass
from typing import Dict, Any
import json


@dataclass
class PowerConfig:
    """
    功耗配置参数
    基于论文优化报告中的实测数据
    """
    # 哨兵模型功耗 (W) - LSTM模型在终端设备上的功耗
    sentinel_power: float = 0.3
    
    # 主模型功耗 (W) - Edge-1DCNN-LSTM在边缘设备上的功耗
    full_power: float = 5.0
    
    # 通信功耗 (W) - 端-边数据传输功耗
    comm_power: float = 0.8
    
    # 空闲功耗 (W) - 设备待机功耗
    idle_power: float = 0.05
    
    def to_dict(self) -> Dict[str, float]:
        return {
            'sentinel_power': self.sentinel_power,
            'full_power': self.full_power,
            'comm_power': self.comm_power,
            'idle_power': self.idle_power
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> 'PowerConfig':
        return cls(**data)


@dataclass
class TimeConfig:
    """
    时间配置参数
    基于Raspberry Pi 4B实测推理时间
    """
    # 哨兵模型推理时间 (s)
    sentinel_inference: float = 0.0218
    
    # 主模型推理时间 (s)
    full_inference: float = 0.0593
    
    # 通信时间 (s)
    comm_time: float = 0.003
    
    # 唤醒时间 (s) - 从休眠到激活的时间
    wakeup_time: float = 0.05
    
    def to_dict(self) -> Dict[str, float]:
        return {
            'sentinel_inference': self.sentinel_inference,
            'full_inference': self.full_inference,
            'comm_time': self.comm_time,
            'wakeup_time': self.wakeup_time
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> 'TimeConfig':
        return cls(**data)


@dataclass
class CooperativeConfig:
    """
    协同推理框架配置
    整合所有配置参数
    """
    # 功耗配置
    power: PowerConfig = None
    
    # 时间配置
    time: TimeConfig = None
    
    # 推理阈值 - 超过此值唤醒主模型
    threshold: float = 0.9
    
    # 模型参数
    input_dim: int = 1
    seq_length: int = 10
    conv_channels: int = 16
    lstm_hidden: int = 32
    lstm_layers: int = 2
    dropout_rate: float = 0.2
    
    # 通信配置
    host: str = '127.0.0.1'
    port: int = 9999
    timeout: float = 5.0
    
    # 数据集配置
    anomaly_ratio: float = 0.15
    num_samples: int = 10000
    
    def __post_init__(self):
        if self.power is None:
            self.power = PowerConfig()
        if self.time is None:
            self.time = TimeConfig()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'power': self.power.to_dict(),
            'time': self.time.to_dict(),
            'threshold': self.threshold,
            'input_dim': self.input_dim,
            'seq_length': self.seq_length,
            'conv_channels': self.conv_channels,
            'lstm_hidden': self.lstm_hidden,
            'lstm_layers': self.lstm_layers,
            'dropout_rate': self.dropout_rate,
            'host': self.host,
            'port': self.port,
            'timeout': self.timeout,
            'anomaly_ratio': self.anomaly_ratio,
            'num_samples': self.num_samples
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CooperativeConfig':
        power_data = data.pop('power', {})
        time_data = data.pop('time', {})
        return cls(
            power=PowerConfig.from_dict(power_data),
            time=TimeConfig.from_dict(time_data),
            **data
        )
    
    def save(self, filepath: str) -> None:
        """保存配置到JSON文件"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load(cls, filepath: str) -> 'CooperativeConfig':
        """从JSON文件加载配置"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls.from_dict(data)


# 预定义配置：最佳优化配置（阈值0.9，能耗节省81.96%）
OPTIMIZED_CONFIG = CooperativeConfig(
    power=PowerConfig(
        sentinel_power=0.3,
        full_power=5.0,
        comm_power=0.8,
        idle_power=0.05
    ),
    time=TimeConfig(
        sentinel_inference=0.0218,
        full_inference=0.0593,
        comm_time=0.003,
        wakeup_time=0.05
    ),
    threshold=0.9,
    input_dim=1,
    seq_length=10,
    conv_channels=16,
    lstm_hidden=32,
    lstm_layers=2,
    dropout_rate=0.2
)


def get_default_config() -> CooperativeConfig:
    """获取默认配置"""
    return OPTIMIZED_CONFIG


if __name__ == "__main__":
    # 测试配置
    config = get_default_config()
    print("=" * 60)
    print("端-边协同推理框架配置")
    print("=" * 60)
    print(f"阈值: {config.threshold}")
    print(f"\n功耗配置:")
    print(f"  哨兵模型: {config.power.sentinel_power} W")
    print(f"  主模型: {config.power.full_power} W")
    print(f"  通信: {config.power.comm_power} W")
    print(f"  空闲: {config.power.idle_power} W")
    print(f"\n时间配置:")
    print(f"  哨兵推理: {config.time.sentinel_inference * 1000:.2f} ms")
    print(f"  主模型推理: {config.time.full_inference * 1000:.2f} ms")
    print(f"  通信时间: {config.time.comm_time * 1000:.2f} ms")
    print(f"  唤醒时间: {config.time.wakeup_time * 1000:.2f} ms")
    print(f"\n目标指标 (阈值={config.threshold}):")
    print(f"  F1分数: ≥ 0.9944")
    print(f"  能耗节省: ≥ 81.96%")
    print(f"  唤醒率: ≤ 8.55%")

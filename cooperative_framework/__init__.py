"""
端-边协同分层推理框架
Edge-End Cooperative Inference Framework

基于论文"Edge-1DCNN-LSTM"第2.4节设计
实现终端哨兵模型与边缘主模型的协同推理
"""

from .config import CooperativeConfig, PowerConfig, TimeConfig
from .cooperative_inference import CooperativeInferenceFramework
from .energy_monitor import EnergyMonitor
from .communication import EdgeCommunication

__version__ = '1.0.0'
__author__ = 'Edge-1DCNN-LSTM Project'

__all__ = [
    'CooperativeInferenceFramework',
    'EnergyMonitor',
    'EdgeCommunication',
    'CooperativeConfig',
    'PowerConfig',
    'TimeConfig'
]

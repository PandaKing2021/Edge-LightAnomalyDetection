# -*- coding: utf-8 -*-
"""
C-MAPSS FD001数据集转换脚本
将原始NASA数据集转换为与模拟数据相同格式的JSON文件，用于1D-CNN-LSTM训练。

输出格式：
{
    "time_sequence_1_value": [sensor_02的数值序列],
    "time_sequence_1_label": [二进制标签序列],
    "time_sequence_2_value": [sensor_03的数值序列],
    "time_sequence_2_label": [二进制标签序列],
    ...
    "time_sequence_5_value": [sensor_08的数值序列],
    "time_sequence_5_label": [二进制标签序列]
}

标签规则：RUL <= 30时为异常(1)，否则正常(0)
"""

import pandas as pd
import numpy as np
import json
import os
import sys
from pathlib import Path

# 添加父目录到路径，以便导入conventor模块
sys.path.append(str(Path(__file__).parent))

def load_cmapss_data(file_path):
    """
    加载C-MAPSS数据集
    
    Args:
        file_path: 数据文件路径
        
    Returns:
        pandas.DataFrame: 包含所有数据的数据框
    """
    # C-MAPSS数据集的列名（根据NASA文档）
    column_names = [
                       'unit_id', 'time_cycles', 'op_setting_1', 'op_setting_2', 'op_setting_3'
                   ] + [f'sensor_{i:02d}' for i in range(1, 22)]
    
    # 读取数据（空格分隔）
    data = pd.read_csv(file_path, sep='\s+', header=None, names=column_names)
    
    return data

def calculate_rul_and_labels(data):
    """
    计算剩余使用寿命(RUL)和故障标签
    
    Args:
        data: 原始数据
        
    Returns:
        tuple: (包含RUL的数据, 故障阈值)
    """
    # 计算每个发动机的最大运行周期
    max_cycles = data.groupby('unit_id')['time_cycles'].max().reset_index()
    max_cycles.columns = ['unit_id', 'max_cycles']
    
    # 合并数据
    data = data.merge(max_cycles, on='unit_id', how='left')
    
    # 计算RUL（剩余使用寿命）
    data['rul'] = data['max_cycles'] - data['time_cycles']
    
    # 定义故障标签（RUL小于等于30个周期为故障）
    failure_threshold = 30
    data['label'] = (data['rul'] <= failure_threshold).astype(int)
    
    return data, failure_threshold

def select_key_sensors():
    """
    选择5个关键传感器用于时间序列生成
    
    基于C-MAPSS数据集的领域知识，选择与发动机性能退化相关性高的传感器：
    1. sensor_02: 风扇入口温度
    2. sensor_03: 低压压缩机出口温度  
    3. sensor_04: 高压压缩机出口压力
    4. sensor_07: 燃料-空气混合比
    5. sensor_08: 核心机转速
    
    Returns:
        list: 5个传感器列名
    """
    return ['sensor_02', 'sensor_03', 'sensor_04', 'sensor_07', 'sensor_08']

def standardize_sensor_sequences(data, sensor_columns):
    """
    对传感器序列进行Z-score标准化
    
    Args:
        data: 包含传感器数据的DataFrame
        sensor_columns: 要标准化的传感器列名列表
        
    Returns:
        tuple: (标准化后的DataFrame, 标准化参数字典)
    """
    data_std = data.copy()
    scaler_params = {}
    
    for sensor in sensor_columns:
        values = data_std[sensor].values.astype(np.float32)
        mean_val = np.mean(values)
        std_val = np.std(values)
        
        # 避免除零
        if std_val < 1e-8:
            std_val = 1.0
            
        # 标准化
        data_std[sensor] = (values - mean_val) / std_val
        
        # 保存参数
        scaler_params[sensor] = {
            'mean': float(mean_val),
            'std': float(std_val)
        }
    
    return data_std, scaler_params

def create_json_dataset(data, sensor_columns, output_file):
    """
    创建JSON格式的数据集
    
    Args:
        data: 包含传感器数据和标签的DataFrame
        sensor_columns: 使用的传感器列名列表
        output_file: 输出JSON文件路径
    """
    # 确保数据按unit_id和time_cycles排序，保持时间顺序
    data = data.sort_values(['unit_id', 'time_cycles']).reset_index(drop=True)
    
    # 提取标签序列
    labels = data['label'].values.tolist()
    
    # 构建输出字典
    dataset = {}
    
    for i, sensor in enumerate(sensor_columns, start=1):
        # 提取传感器值序列
        sensor_values = data[sensor].values.tolist()
        
        # 添加到数据集
        dataset[f'time_sequence_{i}_value'] = sensor_values
        dataset[f'time_sequence_{i}_label'] = labels  # 所有传感器使用相同的标签序列
    
    # 添加元数据
    total_points = len(data)
    num_units = data['unit_id'].nunique()
    positive_ratio = data['label'].mean()
    
    dataset['metadata'] = {
        'dataset': 'C-MAPSS_FD001',
        'total_points': total_points,
        'num_units': num_units,
        'selected_sensors': sensor_columns,
        'failure_threshold': 30,
        'positive_ratio': float(positive_ratio),
        'data_shape': {
            'sequences': 5,
            'points_per_sequence': total_points,
            'labels_per_sequence': total_points
        },
        'description': 'C-MAPSS FD001数据集转换为5个独立传感器时间序列，用于1D-CNN-LSTM异常检测训练'
    }
    
    # 保存为JSON文件
    print(f"保存数据集到: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    
    return dataset

def print_statistics(data, sensor_columns):
    """
    打印数据统计信息
    
    Args:
        data: 数据集
        sensor_columns: 使用的传感器列
    """
    print("=" * 60)
    print("数据集统计信息")
    print("=" * 60)
    print(f"总数据点: {len(data):,}")
    print(f"发动机单元数量: {data['unit_id'].nunique()}")
    print(f"故障比例 (RUL ≤ 30): {data['label'].mean():.3%}")
    print(f"正常点数量: {(data['label'] == 0).sum():,}")
    print(f"异常点数量: {(data['label'] == 1).sum():,}")
    
    print("\n传感器统计:")
    for sensor in sensor_columns:
        values = data[sensor].values
        print(f"  {sensor}: 均值={values.mean():.3f}, 标准差={values.std():.3f}, "
              f"范围=[{values.min():.3f}, {values.max():.3f}]")
    
    print("\n发动机单元运行周期统计:")
    unit_stats = data.groupby('unit_id').agg({
        'time_cycles': ['min', 'max', 'count'],
        'label': 'sum'
    }).round(2)
    unit_stats.columns = ['起始周期', '结束周期', '总周期数', '故障周期数']
    print(unit_stats.head(10))  # 显示前10个发动机
    
    print("=" * 60)

def convert_fd001_to_json():
    """
    主转换函数
    """
    # 路径配置
    project_root = Path(__file__).parent
    input_file = project_root / "datasets" / "NASA" / "train_FD001.txt"
    output_file = project_root / "datasets" / "NASA" / "dataset.json"
    
    # 确保输入文件存在
    if not input_file.exists():
        print(f"错误: 输入文件不存在: {input_file}")
        print("请确保C-MAPSS FD001数据集已放置在正确位置")
        return False
    
    # 创建输出目录（如果不存在）
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # 1. 加载数据
        print("步骤1/5: 加载C-MAPSS FD001数据集...")
        data = load_cmapss_data(input_file)
        print(f"  加载完成: {len(data):,} 行, {data['unit_id'].nunique()} 个发动机单元")
        
        # 2. 计算RUL和标签
        print("步骤2/5: 计算剩余使用寿命(RUL)和故障标签...")
        data, failure_threshold = calculate_rul_and_labels(data)
        print(f"  故障阈值: RUL ≤ {failure_threshold} 周期")
        print(f"  标签分布: 正常={sum(data['label']==0):,}, 异常={sum(data['label']==1):,}")
        
        # 3. 选择关键传感器
        print("步骤3/5: 选择关键传感器...")
        sensor_columns = select_key_sensors()
        print(f"  选择的传感器: {', '.join(sensor_columns)}")
        
        # 4. 标准化传感器数据（可选）
        print("步骤4/5: 标准化传感器数据...")
        data_std, scaler_params = standardize_sensor_sequences(data, sensor_columns)
        
        # 5. 创建JSON数据集
        print("步骤5/5: 创建JSON格式数据集...")
        dataset = create_json_dataset(data_std, sensor_columns, output_file)
        
        # 打印统计信息
        print_statistics(data_std, sensor_columns)
        
        print(f"\n转换成功完成!")
        print(f"输出文件: {output_file}")
        print(f"数据集包含 {len(dataset)-1} 个序列键值对（5个值序列 + 5个标签序列 + 元数据）")
        print(f"每个序列长度: {len(data_std):,} 个数据点")
        
        # 验证输出格式
        print("\n验证输出格式...")
        with open(output_file, 'r', encoding='utf-8') as f:
            loaded_data = json.load(f)
        
        expected_keys = [f'time_sequence_{i}_{part}' 
                        for i in range(1, 6) 
                        for part in ['value', 'label']]
        expected_keys.append('metadata')
        
        for key in expected_keys:
            if key not in loaded_data:
                print(f"  警告: 缺少键 '{key}'")
            else:
                print(f"  [OK] {key}: {len(loaded_data[key])} 个元素")
        
        return True
        
    except Exception as e:
        print(f"转换过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """
    主函数
    """
    print("C-MAPSS FD001 数据集转换工具")
    print("=" * 60)
    print("将原始NASA数据集转换为与模拟数据相同格式的JSON文件")
    print("用于1D-CNN-LSTM异常检测模型训练")
    print("=" * 60)
    
    success = convert_fd001_to_json()
    
    if success:
        print("\n转换成功! 现在可以使用 dataset.json 文件进行模型训练。")
        print("在 ann_train.py 中设置 data_file='datasets/NASA/dataset.json'")
    else:
        print("\n转换失败，请检查错误信息。")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
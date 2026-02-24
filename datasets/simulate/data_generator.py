import numpy as np
import json
import random

# 设置随机种子以确保可重复性
np.random.seed(42)
random.seed(42)

# 每个发生器的正常值范围
generator_ranges = {
    1: {"min": 0, "max": 10},
    2: {"min": 1000, "max": 30000},
    3: {"min": -80, "max": -20},
    4: {"min": -3.9, "max": 82.7},
    5: {"min": -1000, "max": 2000}
}

# 全局参数
seq_length = 50000  # 每个发生器的数据点数量
total_points = seq_length * 5  # 总数据点250000
anomaly_ratio = 0.15  # 异常比例15%
normal_ratio = 0.85  # 正常比例85%

# 异常类型比例（来自文档表3.2）
anomaly_type_ratios = {
    "drift": 0.267,  # 缓慢性漂移
    "spike": 0.240,  # 突发性尖峰
    "offset": 0.253,  # 持续性偏移
    "periodic": 0.240  # 周期性破坏
}

# 计算每个异常类型的目标点数
anomaly_points_per_gen = int(seq_length * anomaly_ratio)  # 7500
drift_points = round(anomaly_type_ratios["drift"] * anomaly_points_per_gen)
spike_points = round(anomaly_type_ratios["spike"] * anomaly_points_per_gen)
offset_points = round(anomaly_type_ratios["offset"] * anomaly_points_per_gen)
periodic_points = anomaly_points_per_gen - drift_points - spike_points - offset_points

# 异常事件长度范围（确保异常数据更加宽泛）
event_length_ranges = {
    "drift": {"min": 200, "max": 1500},  # 延长缓慢性漂移事件
    "spike": {"min": 1, "max": 5},  # 保持尖峰短暂
    "offset": {"min": 300, "max": 1000},  # 延长偏移事件
    "periodic": {"min": 400, "max": 1200}  # 延长周期性破坏事件
}


# 正常数据生成函数（基于文档公式）
def generate_normal_sequence(length, min_val, max_val):
    """生成正常时间序列，包含周期性和噪声。"""
    beta = (min_val + max_val) / 2  # 中点
    A = (max_val - min_val) / 2 * 0.6  # 减小振幅，使数据更集中
    sigma = (max_val - min_val) * 0.03  # 减小噪声

    t = np.arange(length)
    omega = 24  # 周期为24个点

    # 趋势项（轻微趋势）
    alpha = 0.0001 * (max_val - min_val)  # 轻微趋势
    trend = alpha * t

    seasonal = A * np.sin(2 * np.pi * t / omega)  # 周期项
    noise = np.random.normal(0, sigma, length)  # 噪声项

    S_normal = beta + trend + seasonal + noise

    # 裁剪到正常范围内
    S_normal = np.clip(S_normal, min_val, max_val)
    return S_normal


# 异常注入函数（确保异常值更加宽泛）
def inject_anomaly(data, start, length, anomaly_type, min_val, max_val):
    """在指定区间注入异常，确保异常值远离正常范围。"""
    end = start + length
    t_local = np.arange(length)
    W = max_val - min_val  # 正常值范围宽度

    # 计算正常范围的中心和边界
    center = (min_val + max_val) / 2
    range_scale = W * 0.5  # 异常幅度基于正常范围的一半

    if anomaly_type == "drift":
        # 缓慢性漂移: 向正常范围外漂移
        direction = random.choice([-1, 1])  # 随机选择漂移方向
        gamma = direction * 0.15 * W / length  # 漂移率
        drift = gamma * t_local
        data[start:end] += drift

    elif anomaly_type == "spike":
        # 突发性尖峰: 产生明显偏离的尖峰
        direction = random.choice([-1, 1])
        delta = direction * 0.8 * W  # 大幅尖峰
        tau = 1  # 窄尖峰

        # 在事件中点注入尖峰
        t_spike = start + length // 2
        for i in range(max(start, t_spike - 3), min(end, t_spike + 4)):
            spike_val = delta * np.exp(-((i - t_spike) ** 2) / (2 * tau ** 2))
            data[i] += spike_val

    elif anomaly_type == "offset":
        # 持续性偏移: 明显偏离正常值
        direction = random.choice([-1, 1])
        theta = direction * 0.4 * W  # 大幅偏移
        data[start:end] += theta

    elif anomaly_type == "periodic":
        # 周期性破坏: 产生明显的周期性异常
        A_fault = 0.5 * W  # 大幅异常振幅
        omega_fault = 8  # 异常周期（与正常周期不同）
        periodic_anomaly = A_fault * np.sin(2 * np.pi * np.arange(start, end) / omega_fault)
        data[start:end] += periodic_anomaly

    return data  # 不立即裁剪，允许异常值超出正常范围


# 生成异常事件并注入
def add_anomalies(normal_data, min_val, max_val):
    """在正常数据上注入异常事件，确保异常值宽泛分布。"""
    data = normal_data.copy()
    labels = np.zeros(len(normal_data), dtype=int)
    length = len(data)
    events = []

    # 每个异常类型的目标点数
    type_targets = {
        "drift": drift_points,
        "spike": spike_points,
        "offset": offset_points,
        "periodic": periodic_points
    }

    # 为每个异常类型生成事件
    for anomaly_type, target_points in type_targets.items():
        current_points = 0
        max_attempts = 10000
        attempts = 0

        while current_points < target_points and attempts < max_attempts:
            # 随机选择开始点和长度
            start = random.randint(0, length - 1)
            L_min = event_length_ranges[anomaly_type]["min"]
            L_max = event_length_ranges[anomaly_type]["max"]
            L = random.randint(L_min, L_max)
            end = start + L - 1
            if end >= length:
                end = length - 1
                L = end - start + 1

            # 检查是否与已有事件重叠（允许轻微重叠）
            overlap = False
            for event in events:
                # 允许最多10个点的重叠
                if not (end < event["start"] - 10 or start > event["end"] + 10):
                    overlap = True
                    break

            if not overlap:
                points_to_add = min(L, target_points - current_points)
                if points_to_add > 0:
                    L = points_to_add
                    end = start + L - 1
                    if end >= length:
                        end = length - 1
                        L = end - start + 1

                    events.append({"type": anomaly_type, "start": start, "end": end, "length": L})
                    current_points += L

                    # 注入异常
                    data = inject_anomaly(data, start, L, anomaly_type, min_val, max_val)
                    labels[start:end + 1] = 1

            attempts += 1

        if attempts >= max_attempts:
            print(f"警告: {anomaly_type}类型未能达到目标点数，实际生成{current_points}/{target_points}")

    # 最终裁剪到合理范围内（允许超出正常范围20%）
    margin = 0.2 * (max_val - min_val)
    data = np.clip(data, min_val - margin, max_val + margin)

    return data, labels


# 主函数：为每个发生器生成数据
def generate_all_data():
    dataset = {}
    for i in range(1, 6):
        min_val = generator_ranges[i]["min"]
        max_val = generator_ranges[i]["max"]
        print(f"生成发生器 {i} 的数据...")

        # 生成正常序列
        normal_data = generate_normal_sequence(seq_length, min_val, max_val)

        # 注入异常
        data_with_anomalies, labels = add_anomalies(normal_data, min_val, max_val)

        # 存储到dataset
        dataset[f"time_sequence_{i}_value"] = data_with_anomalies.tolist()
        dataset[f"time_sequence_{i}_label"] = labels.tolist()

        # 统计信息
        normal_count = np.sum(labels == 0)
        anomaly_count = np.sum(labels == 1)
        print(f"发生器 {i}: 正常点{normal_count}, 异常点{anomaly_count}, 异常比例{anomaly_count / seq_length:.3f}")

    return dataset


# 生成数据并保存为JSON
if __name__ == "__main__":
    data_dict = generate_all_data()

    with open("dataset.json", "w") as f:
        json.dump(data_dict, f, indent=2)

    print("数据生成完成，已保存到 dataset.json")

    # 验证数据分布
    print("\n数据分布验证:")
    for i in range(1, 6):
        values = data_dict[f"time_sequence_{i}_value"]
        labels = data_dict[f"time_sequence_{i}_label"]

        normal_values = [v for v, l in zip(values, labels) if l == 0]
        anomaly_values = [v for v, l in zip(values, labels) if l == 1]

        print(f"发生器 {i}:")
        print(f"  正常值范围: {min(normal_values):.2f} ~ {max(normal_values):.2f}")
        print(f"  异常值范围: {min(anomaly_values):.2f} ~ {max(anomaly_values):.2f}")
        print(f"  异常比例: {len(anomaly_values) / len(values):.3f}")
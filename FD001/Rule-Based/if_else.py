import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report, roc_curve, precision_recall_curve)
import time
import os
import warnings

warnings.filterwarnings('ignore')


class RuleBasedAnomalyDetector:
    """基于规则的异常检测器"""

    def __init__(self):
        """初始化规则检测器，定义每个发生器的正常值范围"""
        # 根据提供的规则定义正常值范围
        self.rules = {
            1: {'min': -100.0, 'max': 30.0, 'description': '发生器1正常值范围: 30'},
            2: {'min': -100.0, 'max': 30.0, 'description': '发生器2正常值范围: 30'},
            3: {'min': -100.0, 'max': 30.0, 'description': '发生器3正常值范围: 30'},
            4: {'min': -100.0, 'max': 30.7, 'description': '发生器4正常值范围: 30'},
            5: {'min': -100.0, 'max': 30.0, 'description': '发生器5正常值范围: 30'}
        }

        print("规则集异常检测器初始化完成")
        for gen_id, rule in self.rules.items():
            print(f"  发生器{gen_id}: {rule['description']}")

    def detect_anomaly(self, value, generator_id):
        """检测单个值是否为异常"""
        if generator_id not in self.rules:
            raise ValueError(f"未知的发生器ID: {generator_id}")

        rule = self.rules[generator_id]
        is_normal = rule['min'] <= value <= rule['max']
        return 0 if is_normal else 1  # 0表示正常，1表示异常

    def detect_sequence(self, sequence, generator_id):
        """检测整个时间序列"""
        anomalies = []
        for value in sequence:
            anomaly = self.detect_anomaly(value, generator_id)
            anomalies.append(anomaly)
        return np.array(anomalies)

    def evaluate_rules(self, data_file='dataset.json', sequence_length=10):
        """评估规则检测器的性能"""
        print(f"\n{'=' * 60}")
        print("规则集异常检测器评估")
        print(f"{'=' * 60}")

        # 加载数据
        with open(data_file, 'r') as f:
            data = json.load(f)

        all_predictions = []
        all_labels = []
        inference_times = []

        # 对每个发生器进行评估
        for generator_id in range(1, 6):
            values_key = f"time_sequence_{generator_id}_value"
            labels_key = f"time_sequence_{generator_id}_label"

            if values_key not in data or labels_key not in data:
                print(f"警告: 发生器{generator_id}的数据不存在，跳过")
                continue

            values = np.array(data[values_key])
            labels = np.array(data[labels_key])

            if len(values) == 0:
                print(f"警告: 发生器{generator_id}的数据为空，跳过")
                continue

            print(f"\n评估发生器{generator_id}:")
            print(f"  数据长度: {len(values)}")
            print(f"  正常值范围: [{self.rules[generator_id]['min']}, {self.rules[generator_id]['max']}]")

            # 检测异常
            start_time = time.time()
            predictions = self.detect_sequence(values, generator_id)
            inference_time = time.time() - start_time

            # 确保预测和标签长度一致
            min_len = min(len(predictions), len(labels))
            predictions = predictions[:min_len]
            labels = labels[:min_len]

            all_predictions.extend(predictions)
            all_labels.extend(labels)
            inference_times.append(inference_time)

            # 计算每个发生器的指标
            if len(np.unique(labels)) > 1:
                accuracy = accuracy_score(labels, predictions)
                precision = precision_score(labels, predictions, zero_division=0)
                recall = recall_score(labels, predictions, zero_division=0)
                f1 = f1_score(labels, predictions, zero_division=0)

                print(f"  准确率: {accuracy:.4f}")
                print(f"  精确率: {precision:.4f}")
                print(f"  召回率: {recall:.4f}")
                print(f"  F1分数: {f1:.4f}")
                print(f"  推理时间: {inference_time:.4f}秒")

        # 总体评估
        if len(all_predictions) > 0:
            print(f"\n{'=' * 60}")
            print("总体评估结果:")
            print(f"{'=' * 60}")

            accuracy = accuracy_score(all_labels, all_predictions)
            precision = precision_score(all_labels, all_predictions, zero_division=0)
            recall = recall_score(all_labels, all_predictions, zero_division=0)
            f1 = f1_score(all_labels, all_predictions, zero_division=0)

            # 混淆矩阵
            cm = confusion_matrix(all_labels, all_predictions)
            if cm.size == 4:
                tn, fp, fn, tp = cm.ravel()
            else:
                tn, fp, fn, tp = 0, 0, 0, 0

            # 详细指标
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

            print(f"总样本数: {len(all_predictions)}")
            print(f"准确率: {accuracy:.4f}")
            print(f"精确率: {precision:.4f}")
            print(f"召回率: {recall:.4f}")
            print(f"F1分数: {f1:.4f}")
            print(f"特异度: {specificity:.4f}")
            print(f"假正率: {fpr:.4f}")
            print(f"假负率: {fnr:.4f}")

            print(f"\n混淆矩阵:")
            print(f"TP: {tp}, FP: {fp}")
            print(f"FN: {fn}, TN: {tn}")

            print(f"\n详细分类报告:")
            print(classification_report(all_labels, all_predictions,
                                        target_names=['正常', '异常'], digits=4))

            # 可视化混淆矩阵
            self.plot_confusion_matrix(cm, "规则集检测器混淆矩阵")

            # 保存结果
            results = {
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1),
                'specificity': float(specificity),
                'fpr': float(fpr),
                'fnr': float(fnr),
                'confusion_matrix': cm.tolist(),
                'total_samples': len(all_predictions),
                'total_time': sum(inference_times)
            }

            with open('rule_based_results.json', 'w') as f:
                json.dump(results, f, indent=2)

            print(f"\n结果已保存到: rule_based_results.json")

            return results
        else:
            print("错误: 没有有效的数据进行评估")
            return None

    def plot_confusion_matrix(self, cm, title="混淆矩阵"):
        """绘制混淆矩阵"""
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(title, fontsize=14)
        plt.colorbar()

        tick_marks = np.arange(2)
        plt.xticks(tick_marks, ['预测正常', '预测异常'])
        plt.yticks(tick_marks, ['真实正常', '真实异常'])

        # 在格子中显示数字
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('真实标签', fontsize=12)
        plt.xlabel('预测标签', fontsize=12)
        plt.savefig('rule_based_confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("混淆矩阵已保存到: rule_based_confusion_matrix.png")

    def analyze_value_distribution(self, data_file='dataset.json'):
        """分析每个发生器的数值分布"""
        print(f"\n{'=' * 60}")
        print("数值分布分析")
        print(f"{'=' * 60}")

        with open(data_file, 'r') as f:
            data = json.load(f)

        plt.figure(figsize=(12, 8))

        for generator_id in range(1, 6):
            values_key = f"time_sequence_{generator_id}_value"

            if values_key not in data:
                continue

            values = np.array(data[values_key])
            rule = self.rules[generator_id]

            # 创建子图
            plt.subplot(2, 3, generator_id)

            # 绘制直方图
            plt.hist(values, bins=30, alpha=0.7, color='blue', edgecolor='black')

            # 标记正常值范围
            plt.axvline(x=rule['min'], color='red', linestyle='--', linewidth=2,
                        label=f'下限: {rule["min"]}')
            plt.axvline(x=rule['max'], color='green', linestyle='--', linewidth=2,
                        label=f'上限: {rule["max"]}')

            # 标记异常值比例
            predictions = self.detect_sequence(values, generator_id)
            anomaly_ratio = np.mean(predictions) * 100

            plt.title(f'发生器{generator_id} (异常率: {anomaly_ratio:.1f}%)', fontsize=10)
            plt.xlabel('数值', fontsize=8)
            plt.ylabel('频数', fontsize=8)
            plt.legend(fontsize=8)
            plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('value_distribution_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("数值分布图已保存到: value_distribution_analysis.png")


def main():
    """主函数"""
    # 检查数据文件是否存在
    data_file = 'test.json'
    if not os.path.exists(data_file):
        print(f"错误: 数据文件 {data_file} 不存在")
        print("请确保dataset.json文件在当前目录")
        return

    # 创建规则检测器
    detector = RuleBasedAnomalyDetector()

    # 分析数值分布
    detector.analyze_value_distribution(data_file)

    # 评估规则检测器
    results = detector.evaluate_rules(data_file)

    if results:
        print(f"\n规则集检测器评估完成!")
        print(f"总体F1分数: {results['f1_score']:.4f}")


if __name__ == "__main__":
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    main()

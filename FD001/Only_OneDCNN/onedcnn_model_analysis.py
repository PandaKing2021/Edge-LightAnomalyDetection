import torch
import torch.nn as nn
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report, roc_curve, precision_recall_curve,
                             auc)
from torch.utils.data import Dataset, DataLoader
import time
import os
import warnings

warnings.filterwarnings('ignore')



class InferenceDataset(Dataset):
    """推理测试数据集类（适配原始数据生成器格式）"""

    def __init__(self, data_file, sequence_length=10, generator_id=1, normalize=True):
        """
        初始化推理数据集

        Args:
            data_file: JSON数据文件路径（原始数据生成器格式）
            sequence_length: 序列长度（需与训练时一致）
            generator_id: 发生器ID (1-5)，默认使用第一个发生器
            normalize: 是否进行标准化（需与训练时一致）
        """
        self.sequence_length = sequence_length
        self.generator_id = generator_id
        self.normalize = normalize

        # 加载推理数据
        print(f"加载推理数据集: {data_file}")
        with open(data_file, 'r') as f:
            self.data = json.load(f)

        # 检查数据格式并获取数据
        self.values, self.labels = self._load_data()

        # 数据预处理
        if normalize:
            self.values = self._normalize_data(self.values)

        # 创建序列
        self.samples, self.sample_labels = self._create_sequences()

        print(f"数据加载完成: {len(self.samples)} 个样本")
        print(f"正样本比例: {np.mean(self.sample_labels):.3f}")

    def _load_data(self):
        """加载数据（适配原始数据生成器格式）"""
        # 尝试原始数据生成器格式
        values_key = f"time_sequence_{self.generator_id}_value"
        labels_key = f"time_sequence_{self.generator_id}_label"

        if values_key in self.data and labels_key in self.data:
            print(f"使用数据生成器格式: {values_key}, {labels_key}")
            values = np.array(self.data[values_key], dtype=np.float32)
            labels = np.array(self.data[labels_key], dtype=np.float32)
        # 尝试简单格式（兼容旧版本）
        elif 'values' in self.data and 'labels' in self.data:
            print("使用简单格式: values, labels")
            values = np.array(self.data['values'], dtype=np.float32)
            labels = np.array(self.data['labels'], dtype=np.float32)
        else:
            # 尝试寻找其他可能的键
            available_keys = list(self.data.keys())
            print(f"可用数据键: {available_keys}")

            # 尝试自动检测格式
            value_keys = [k for k in available_keys if 'value' in k.lower()]
            label_keys = [k for k in available_keys if 'label' in k.lower()]

            if value_keys and label_keys:
                print(f"自动检测到值键: {value_keys[0]}, 标签键: {label_keys[0]}")
                values = np.array(self.data[value_keys[0]], dtype=np.float32)
                labels = np.array(self.data[label_keys[0]], dtype=np.float32)
            else:
                raise ValueError(
                    f"数据格式错误: 无法识别数据格式。期望的数据生成器格式包含 '{values_key}' 和 '{labels_key}' 键")

        if len(values) != len(labels):
            raise ValueError(f"数据长度不匹配: values({len(values)}) != labels({len(labels)})")

        print(f"数据维度: values={values.shape}, labels={labels.shape}")
        return values, labels

    def _normalize_data(self, values):
        """标准化数据"""
        mean = np.mean(values)
        std = np.std(values)
        normalized = (values - mean) / std
        print(f"数据标准化: mean={mean:.4f}, std={std:.4f}")
        return normalized

    def _create_sequences(self):
        """创建序列"""
        sequences = []
        sequence_labels = []

        # 检查数据长度是否足够
        if len(self.values) < self.sequence_length:
            raise ValueError(f"数据长度({len(self.values)})小于序列长度({self.sequence_length})")

        # 创建滑动窗口
        for i in range(len(self.values) - self.sequence_length + 1):
            seq = self.values[i:i + self.sequence_length]
            label = self.labels[i + self.sequence_length - 1]

            sequences.append(seq)
            sequence_labels.append(label)

        return np.array(sequences, dtype=np.float32), np.array(sequence_labels, dtype=np.float32)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        label = self.sample_labels[idx]

        # 转换为PyTorch张量，添加特征维度
        sample_tensor = torch.FloatTensor(sample).unsqueeze(-1)  # 形状: [sequence_length, 1]
        label_tensor = torch.FloatTensor([label])

        return sample_tensor, label_tensor


class ModelAnalyzer:
    """模型推理分析器"""

    def __init__(self, model_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = torch.device(device)
        self.model_path = model_path
        self.model = None
        self.config = None
        self.results = {}

        # 加载模型
        self._load_model()

    def _load_model(self):
        """加载训练好的模型"""
        try:
            print(f"加载模型: {self.model_path}")
            checkpoint = torch.load(self.model_path, map_location=self.device)

            # 获取模型配置
            if 'config' in checkpoint:
                self.config = checkpoint['config'].get('model_config', {})
            else:
                # 默认配置（应与训练时一致）
                self.config = {
                    'input_dim': 1,
                    'seq_length': 10,
                    'conv_channels': 16,
                    'dropout_rate': 0.2
                }

            print(f"模型配置: {self.config}")

            # 创建模型实例
            self.model = self._create_model()

            # 加载权重
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)

            self.model.to(self.device)
            self.model.eval()
            print(f"模型加载成功，设备: {self.device}")

        except Exception as e:
            print(f"模型加载失败: {e}")
            raise

    def _create_model(self):
        """创建模型实例"""

        class Pure1DCNN(nn.Module):
            def __init__(self, input_dim=1, seq_length=10, conv_channels=16, dropout_rate=0.2):
                super(Pure1DCNN, self).__init__()
                self.conv1d = nn.Conv1d(input_dim, conv_channels, kernel_size=3, padding=1)
                self.batchnorm = nn.BatchNorm1d(conv_channels)
                self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
                self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
                self.dropout = nn.Dropout(dropout_rate)
                self.fc = nn.Linear(conv_channels, 1)

            def forward(self, x):
                x = x.transpose(1, 2)
                x = self.conv1d(x)
                x = torch.relu(x)
                x = self.batchnorm(x)
                x = self.pool(x)
                x = self.global_avg_pool(x)
                x = x.squeeze(-1)
                x = self.dropout(x)
                x = self.fc(x)
                return torch.sigmoid(x)

        return Pure1DCNN(**self.config)

    def inference(self, data_loader, threshold=0.5):
        """执行推理"""
        all_predictions = []
        all_probabilities = []
        all_labels = []
        inference_times = []

        print("开始推理...")

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(data_loader):
                data, target = data.to(self.device), target.to(self.device)

                # 计时推理
                start_time = time.perf_counter()
                output = self.model(data)
                inference_time = time.perf_counter() - start_time

                probabilities = output.cpu().numpy().flatten()
                predictions = (probabilities > threshold).astype(int)

                all_probabilities.extend(probabilities)
                all_predictions.extend(predictions)
                all_labels.extend(target.cpu().numpy().flatten())
                inference_times.append(inference_time)

                if (batch_idx + 1) % 10 == 0:
                    print(f"已处理 {batch_idx + 1}/{len(data_loader)} 批次")

        total_samples = len(all_labels)
        total_time = sum(inference_times)

        results = {
            'predictions': np.array(all_predictions),
            'probabilities': np.array(all_probabilities),
            'labels': np.array(all_labels),
            'inference_times': inference_times,
            'total_samples': total_samples,
            'total_time': total_time,
            'avg_time_per_sample': total_time / total_samples * 1000,  # 毫秒
            'throughput': total_samples / total_time  # 样本/秒
        }

        print(f"推理完成: {total_samples} 个样本, 总时间: {total_time:.2f} 秒")
        print(f"平均推理时间: {results['avg_time_per_sample']:.2f} ms/样本")
        print(f"吞吐量: {results['throughput']:.2f} 样本/秒")

        return results

    def calculate_metrics(self, predictions, labels, probabilities):
        """计算评估指标"""
        # 基础指标
        accuracy = accuracy_score(labels, predictions)
        precision = precision_score(labels, predictions, zero_division=0)
        recall = recall_score(labels, predictions, zero_division=0)
        f1 = f1_score(labels, predictions, zero_division=0)

        # AUC
        if len(np.unique(labels)) > 1:
            try:
                auc_score = roc_auc_score(labels, probabilities)
            except:
                auc_score = 0.0
        else:
            auc_score = 0.0

        # 混淆矩阵
        cm = confusion_matrix(labels, predictions)
        if cm.size == 4:
            tn, fp, fn, tp = cm.ravel()
        else:
            tn, fp, fn, tp = 0, 0, 0, 0

        # 详细指标
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0  # 阴性预测值
        prevalence = (tp + fn) / (tp + fp + tn + fn)  # 患病率

        # F1分数的变体
        f2_score_value = (5 * precision * recall) / (4 * precision + recall + 1e-8) if (precision + recall) > 0 else 0

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'f2_score': f2_score_value,
            'auc': auc_score,
            'specificity': specificity,
            'fpr': fpr,
            'fnr': fnr,
            'npv': npv,
            'prevalence': prevalence,
            'confusion_matrix': cm.tolist(),
            'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn)
        }

    def analyze_threshold(self, probabilities, labels, thresholds=None):
        """分析不同阈值下的性能"""
        if thresholds is None:
            thresholds = np.arange(0.1, 1.0, 0.05)

        threshold_analysis = {}

        for threshold in thresholds:
            predictions = (probabilities > threshold).astype(int)
            metrics = self.calculate_metrics(predictions, labels, probabilities)
            threshold_analysis[threshold] = metrics

        return threshold_analysis

    def find_optimal_threshold(self, probabilities, labels, metric='f1'):
        """寻找最佳阈值"""
        thresholds = np.arange(0.01, 1.0, 0.01)
        best_threshold = 0.5
        best_metric_value = 0

        for threshold in thresholds:
            predictions = (probabilities > threshold).astype(int)
            metrics = self.calculate_metrics(predictions, labels, probabilities)

            metric_value = metrics.get(metric, 0)
            if metric_value > best_metric_value:
                best_metric_value = metric_value
                best_threshold = threshold

        return best_threshold, best_metric_value

    def plot_confusion_matrix(self, cm, title="混淆矩阵", save_path=None):
        """绘制混淆矩阵"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['预测正常', '预测故障'],
                    yticklabels=['真实正常', '真实故障'])
        plt.title(title, fontsize=14)
        plt.ylabel('真实标签', fontsize=12)
        plt.xlabel('预测标签', fontsize=12)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"混淆矩阵已保存: {save_path}")
        plt.show()

    def plot_roc_curve(self, labels, probabilities, title="ROC曲线", save_path=None):
        """绘制ROC曲线"""
        if len(np.unique(labels)) > 1:
            fpr, tpr, thresholds = roc_curve(labels, probabilities)
            auc_score = roc_auc_score(labels, probabilities)

            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, label=f'ROC曲线 (AUC = {auc_score:.4f})', linewidth=2)
            plt.plot([0, 1], [0, 1], 'k--', label='随机分类器', linewidth=1, alpha=0.5)
            plt.fill_between(fpr, tpr, alpha=0.2)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('假正率 (FPR)', fontsize=12)
            plt.ylabel('真正率 (TPR)', fontsize=12)
            plt.title(title, fontsize=14)
            plt.legend(loc='lower right')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"ROC曲线已保存: {save_path}")
            plt.show()

            return auc_score, fpr, tpr, thresholds
        else:
            print("警告: 数据中只有一个类别，无法绘制ROC曲线")
            return 0.0, None, None, None

    def plot_precision_recall_curve(self, labels, probabilities, title="精确率-召回率曲线", save_path=None):
        """绘制精确率-召回率曲线"""
        precision, recall, thresholds = precision_recall_curve(labels, probabilities)

        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, label='P-R曲线', linewidth=2)
        plt.fill_between(recall, precision, alpha=0.2)
        plt.xlabel('召回率 (Recall)', fontsize=12)
        plt.ylabel('精确率 (Precision)', fontsize=12)
        plt.title(title, fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"P-R曲线已保存: {save_path}")
        plt.show()

        return precision, recall, thresholds

    def plot_threshold_analysis(self, threshold_analysis, save_path=None):
        """绘制阈值分析图"""
        thresholds = list(threshold_analysis.keys())
        f1_scores = [threshold_analysis[t]['f1_score'] for t in thresholds]
        precisions = [threshold_analysis[t]['precision'] for t in thresholds]
        recalls = [threshold_analysis[t]['recall'] for t in thresholds]

        plt.figure(figsize=(10, 6))
        plt.plot(thresholds, f1_scores, 'b-', label='F1分数', linewidth=2)
        plt.plot(thresholds, precisions, 'r-', label='精确率', linewidth=2)
        plt.plot(thresholds, recalls, 'g-', label='召回率', linewidth=2)
        plt.xlabel('分类阈值', fontsize=12)
        plt.ylabel('分数', fontsize=12)
        plt.title('阈值敏感性分析', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"阈值分析图已保存: {save_path}")
        plt.show()

        # 找到最佳阈值
        best_idx = np.argmax(f1_scores)
        best_threshold = thresholds[best_idx]
        best_f1 = f1_scores[best_idx]

        print(f"最佳阈值: {best_threshold:.2f} (F1分数: {best_f1:.4f})")
        return best_threshold

    def plot_probability_distribution(self, probabilities, labels, title="预测概率分布", save_path=None):
        """绘制预测概率分布"""
        pred_normal = probabilities[labels == 0]
        pred_fault = probabilities[labels == 1]

        plt.figure(figsize=(10, 6))

        if len(pred_normal) > 0:
            plt.hist(pred_normal, bins=20, alpha=0.7, label='正常样本', color='green')
        if len(pred_fault) > 0:
            plt.hist(pred_fault, bins=20, alpha=0.7, label='故障样本', color='red')

        plt.axvline(x=0.5, color='blue', linestyle='--', label='阈值(0.5)')
        plt.xlabel('预测概率', fontsize=12)
        plt.ylabel('样本数量', fontsize=12)
        plt.title(title, fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"概率分布图已保存: {save_path}")
        plt.show()

    def generate_detailed_report(self, results, dataset_name="推理数据集"):
        """生成详细分析报告"""
        metrics = self.calculate_metrics(
            results['predictions'],
            results['labels'],
            results['probabilities']
        )

        print(f"\n{'=' * 80}")
        print(f"仅1D-CNN模型推理分析报告 - {dataset_name}")
        print(f"{'=' * 80}")

        # 数据集统计
        print(f"\n📊 数据集统计:")
        print(f"  总样本数: {results['total_samples']}")
        print(f"  正样本数: {np.sum(results['labels'])}")
        print(f"  负样本数: {len(results['labels']) - np.sum(results['labels'])}")
        print(f"  正样本比例: {np.mean(results['labels']):.3f}")

        # 性能指标
        print(f"\n📈 性能指标 (阈值=0.5):")
        print(f"  准确率 (Accuracy): {metrics['accuracy']:.4f}")
        print(f"  精确率 (Precision): {metrics['precision']:.4f}")
        print(f"  召回率 (Recall): {metrics['recall']:.4f}")
        print(f"  F1分数: {metrics['f1_score']:.4f}")
        print(f"  F2分数: {metrics['f2_score']:.4f}")
        print(f"  AUC: {metrics['auc']:.4f}")
        print(f"  特异度 (Specificity): {metrics['specificity']:.4f}")
        print(f"  阴性预测值 (NPV): {metrics['npv']:.4f}")

        # 错误率
        print(f"\n⚠️  错误分析:")
        print(f"  假正率 (FPR): {metrics['fpr']:.4f}")
        print(f"  假负率 (FNR): {metrics['fnr']:.4f}")

        # 推理性能
        print(f"\n⚡ 推理性能:")
        print(f"  总推理时间: {results['total_time']:.2f} 秒")
        print(f"  平均推理时间: {results['avg_time_per_sample']:.2f} 毫秒/样本")
        print(f"  吞吐量: {results['throughput']:.2f} 样本/秒")

        # 混淆矩阵
        print(f"\n📊 混淆矩阵:")
        cm = np.array(metrics['confusion_matrix'])
        print(f"         预测正常  预测故障")
        print(f"真实正常   {cm[0][0]:6d}    {cm[0][1]:6d}")
        print(f"真实故障   {cm[1][0]:6d}    {cm[1][1]:6d}")

        # 详细分类报告
        print(f"\n📋 详细分类报告:")
        print(classification_report(results['labels'], results['predictions'],
                                    target_names=['正常', '故障'], digits=4))

        # 保存结果
        report_data = {
            'dataset_name': dataset_name,
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'model_path': self.model_path,
            'model_config': self.config,
            'dataset_statistics': {
                'total_samples': int(results['total_samples']),
                'positive_samples': int(np.sum(results['labels'])),
                'negative_samples': int(len(results['labels']) - np.sum(results['labels'])),
                'positive_ratio': float(np.mean(results['labels']))
            },
            'performance_metrics': metrics,
            'inference_performance': {
                'total_time': float(results['total_time']),
                'avg_time_per_sample': float(results['avg_time_per_sample']),
                'throughput': float(results['throughput'])
            },
            'predictions': results['predictions'].tolist(),
            'probabilities': results['probabilities'].tolist(),
            'labels': results['labels'].tolist()
        }

        return report_data, metrics

    def run_analysis(self, data_file, batch_size=64, threshold=0.5,
                     save_plots=True, output_dir='results', generator_id=1):
        """运行完整分析流程"""

        # 创建输出目录
        if save_plots and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 加载推理数据集
        dataset = InferenceDataset(data_file,
                                   sequence_length=self.config['seq_length'],
                                   generator_id=generator_id)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        print(f"\n{'=' * 80}")
        print("开始推理测试分析")
        print(f"数据集: {data_file}")
        print(f"批次大小: {batch_size}")
        print(f"分类阈值: {threshold}")
        print(f"发生器ID: {generator_id}")
        print(f"{'=' * 80}")

        # 执行推理
        results = self.inference(data_loader, threshold=threshold)

        # 生成详细报告
        report_data, metrics = self.generate_detailed_report(results, "推理测试集")

        # 可视化分析
        if save_plots:
            print(f"\n📊 生成可视化图表...")

            # 混淆矩阵
            cm = np.array(metrics['confusion_matrix'])
            self.plot_confusion_matrix(
                cm,
                title="仅1D-CNN模型混淆矩阵",
                save_path=os.path.join(output_dir, 'confusion_matrix.png')
            )

            # ROC曲线
            if len(np.unique(results['labels'])) > 1:
                self.plot_roc_curve(
                    results['labels'], results['probabilities'],
                    title="仅1D-CNN模型ROC曲线",
                    save_path=os.path.join(output_dir, 'roc_curve.png')
                )

            # 精确率-召回率曲线
            self.plot_precision_recall_curve(
                results['labels'], results['probabilities'],
                title="仅1D-CNN模型精确率-召回率曲线",
                save_path=os.path.join(output_dir, 'pr_curve.png')
            )

            # 概率分布
            self.plot_probability_distribution(
                results['probabilities'], results['labels'],
                title="仅1D-CNN模型预测概率分布",
                save_path=os.path.join(output_dir, 'probability_distribution.png')
            )

            # 阈值分析
            threshold_analysis = self.analyze_threshold(
                results['probabilities'], results['labels']
            )
            best_threshold = self.plot_threshold_analysis(
                threshold_analysis,
                save_path=os.path.join(output_dir, 'threshold_analysis.png')
            )

            report_data['optimal_threshold'] = float(best_threshold)
            report_data['generator_id'] = generator_id

        # 保存报告
        report_file = os.path.join(output_dir, 'inference_report.json')
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)

        print(f"\n✅ 分析完成!")
        print(f"报告已保存: {report_file}")

        if save_plots:
            print(f"图表已保存到: {output_dir}/")

        return report_data, results


def main():
    """主函数 - 执行推理分析"""

    # 配置参数
    MODEL_PATH = "pure_1dcnn_best.pth"  # 训练好的模型路径
    INFERENCE_DATA_FILE = "test.json"  # 推理数据集路径
    BATCH_SIZE = 64  # 批次大小
    THRESHOLD = 0.5  # 分类阈值
    GENERATOR_ID = 1  # 发生器ID
    SAVE_PLOTS = True  # 是否保存图表
    OUTPUT_DIR = "inference_results"  # 输出目录

    # 检查文件是否存在
    if not os.path.exists(MODEL_PATH):
        print(f"❌ 错误: 模型文件 {MODEL_PATH} 不存在")
        print("请先训练模型或检查模型路径")
        return

    if not os.path.exists(INFERENCE_DATA_FILE):
        print(f"❌ 错误: 推理数据文件 {INFERENCE_DATA_FILE} 不存在")
        print("请检查数据文件路径")
        return

    try:
        # 初始化分析器
        analyzer = ModelAnalyzer(MODEL_PATH)

        # 运行完整分析
        report, results = analyzer.run_analysis(
            data_file=INFERENCE_DATA_FILE,
            batch_size=BATCH_SIZE,
            threshold=THRESHOLD,
            save_plots=SAVE_PLOTS,
            output_dir=OUTPUT_DIR,
            generator_id=GENERATOR_ID
        )

        # 打印最佳阈值建议
        best_threshold, best_f1 = analyzer.find_optimal_threshold(
            results['probabilities'], results['labels'], metric='f1'
        )

        print(f"\n💡 最佳阈值建议:")
        print(f"  基于F1分数的最佳阈值: {best_threshold:.3f}")
        print(f"  对应的F1分数: {best_f1:.4f}")

        # 使用最佳阈值重新评估
        if best_threshold != THRESHOLD:
            print(f"\n使用最佳阈值 {best_threshold:.3f} 重新评估:")
            predictions_best = (results['probabilities'] > best_threshold).astype(int)
            metrics_best = analyzer.calculate_metrics(
                predictions_best, results['labels'], results['probabilities']
            )
            print(
                f"  F1分数: {metrics_best['f1_score']:.4f} (原始阈值: {report['performance_metrics']['f1_score']:.4f})")
            print(
                f"  精确率: {metrics_best['precision']:.4f} (原始阈值: {report['performance_metrics']['precision']:.4f})")
            print(f"  召回率: {metrics_best['recall']:.4f} (原始阈值: {report['performance_metrics']['recall']:.4f})")

    except Exception as e:
        print(f"❌ 分析过程中发生错误: {e}")
        import traceback
        traceback.print_exc()


def test_dataset_format():
    """测试数据集格式识别功能"""
    # 创建测试数据
    test_data = {
        "time_sequence_1_value": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
        "time_sequence_1_label": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
        "time_sequence_2_value": [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0],
        "time_sequence_2_label": [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
    }

    # 保存测试数据
    test_file = "test_dataset.json"
    with open(test_file, 'w') as f:
        json.dump(test_data, f)

    print(f"创建测试文件: {test_file}")

    try:
        # 测试数据加载
        dataset = InferenceDataset(test_file, sequence_length=10, generator_id=1)
        print(f"测试数据集加载成功: {len(dataset)} 个样本")

        # 测试数据加载器
        data_loader = DataLoader(dataset, batch_size=2, shuffle=False)
        print(f"数据加载器创建成功: {len(data_loader)} 个批次")

        # 测试第一个批次
        for batch_idx, (data, target) in enumerate(data_loader):
            print(f"批次 {batch_idx}: 数据形状={data.shape}, 标签形状={target.shape}")
            if batch_idx >= 1:
                break

        # 清理测试文件
        os.remove(test_file)
        print(f"清理测试文件: {test_file}")

    except Exception as e:
        print(f"测试失败: {e}")
        if os.path.exists(test_file):
            os.remove(test_file)


if __name__ == "__main__":
    # 设置matplotlib中文字体（可选）
    try:
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
    except:
        print("警告: 字体设置失败，使用默认字体")

    # 测试数据集格式识别
    print("测试数据集格式识别...")
    test_dataset_format()

    print("\n" + "=" * 80)
    print("开始正式推理分析")
    print("=" * 80)

    # 运行主函数
    main()
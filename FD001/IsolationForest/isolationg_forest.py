import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report, roc_curve, precision_recall_curve)
from sklearn.preprocessing import StandardScaler
import time
import os
import warnings

warnings.filterwarnings('ignore')


class IsolationForestAnomalyDetector:
    """孤立森林异常检测器"""

    def __init__(self, contamination=0.1, n_estimators=100, random_state=42):
        """
        初始化孤立森林检测器

        Args:
            contamination: 异常值比例估计
            n_estimators: 树的数量
            random_state: 随机种子
        """
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.random_state = random_state

        # 创建孤立森林模型
        self.model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state,
            n_jobs=-1  # 使用所有CPU核心
        )

        # 数据标准化器
        self.scaler = StandardScaler()

        print(f"孤立森林检测器初始化完成")
        print(f"  树的数量: {n_estimators}")
        print(f"  异常比例估计: {contamination}")
        print(f"  随机种子: {random_state}")

    def prepare_data(self, data_file='test.json'):
        """准备训练和测试数据"""
        with open(data_file, 'r') as f:
            data = json.load(f)

        all_values = []
        all_labels = []
        generator_ids = []

        # 收集所有发生器的数据
        for generator_id in range(1, 6):
            values_key = f"time_sequence_{generator_id}_value"
            labels_key = f"time_sequence_{generator_id}_label"

            if values_key in data and labels_key in data:
                values = np.array(data[values_key]).reshape(-1, 1)
                labels = np.array(data[labels_key])

                all_values.append(values)
                all_labels.append(labels)
                generator_ids.extend([generator_id] * len(values))

        if len(all_values) == 0:
            raise ValueError("没有找到有效数据")

        # 合并数据
        X = np.vstack(all_values)
        y = np.hstack(all_labels)
        gen_ids = np.array(generator_ids)

        print(f"数据准备完成:")
        print(f"  总样本数: {len(X)}")
        print(f"  异常比例: {np.mean(y):.3f}")

        return X, y, gen_ids

    def train_and_evaluate(self, data_file='test.json', test_size=0.3):
        """训练并评估模型"""
        print(f"\n{'=' * 60}")
        print("孤立森林异常检测器训练与评估")
        print(f"{'=' * 60}")

        # 准备数据
        X, y, gen_ids = self.prepare_data(data_file)

        # 数据标准化
        X_scaled = self.scaler.fit_transform(X)

        # 划分训练集和测试集
        n_samples = len(X_scaled)
        split_idx = int(n_samples * (1 - test_size))

        indices = np.random.permutation(n_samples)
        train_idx = indices[:split_idx]
        test_idx = indices[split_idx:]

        X_train = X_scaled[train_idx]
        y_train = y[train_idx]
        X_test = X_scaled[test_idx]
        y_test = y[test_idx]
        gen_ids_test = gen_ids[test_idx]

        print(f"\n数据集划分:")
        print(f"  训练集: {len(X_train)} 样本")
        print(f"  测试集: {len(X_test)} 样本")
        print(f"  训练集异常比例: {np.mean(y_train):.3f}")
        print(f"  测试集异常比例: {np.mean(y_test):.3f}")

        # 训练模型
        print("\n开始训练孤立森林模型...")
        start_time = time.time()
        self.model.fit(X_train)
        train_time = time.time() - start_time
        print(f"模型训练完成，耗时: {train_time:.2f}秒")

        # 预测
        print("\n开始预测...")
        start_time = time.time()

        # 训练集预测
        train_pred = self.model.predict(X_train)
        # 孤立森林返回1表示正常，-1表示异常，转换为0/1格式
        train_pred_binary = np.where(train_pred == 1, 0, 1)

        # 测试集预测
        test_pred = self.model.predict(X_test)
        test_pred_binary = np.where(test_pred == 1, 0, 1)

        predict_time = time.time() - start_time
        print(f"预测完成，耗时: {predict_time:.2f}秒")

        # 评估训练集
        print(f"\n{'=' * 40}")
        print("训练集评估结果")
        print(f"{'=' * 40}")
        train_metrics = self._calculate_metrics(y_train, train_pred_binary)

        # 评估测试集
        print(f"\n{'=' * 40}")
        print("测试集评估结果")
        print(f"{'=' * 40}")
        test_metrics = self._calculate_metrics(y_test, test_pred_binary)

        # 按发生器分析
        print(f"\n{'=' * 40}")
        print("按发生器分析结果")
        print(f"{'=' * 40}")
        generator_results = {}
        for gen_id in range(1, 6):
            gen_mask = gen_ids_test == gen_id
            if np.any(gen_mask):
                y_gen = y_test[gen_mask]
                pred_gen = test_pred_binary[gen_mask]

                if len(np.unique(y_gen)) > 1:
                    accuracy = accuracy_score(y_gen, pred_gen)
                    f1 = f1_score(y_gen, pred_gen, zero_division=0)
                    generator_results[gen_id] = {
                        'accuracy': accuracy,
                        'f1_score': f1,
                        'samples': len(y_gen)
                    }
                    print(f"发生器{gen_id}: 准确率={accuracy:.4f}, F1={f1:.4f}, 样本数={len(y_gen)}")
                else:
                    print(f"发生器{gen_id}: 数据中只有一个类别，无法计算F1分数")

        # 可视化
        self._visualize_results(X_test, y_test, test_pred_binary, X_scaled)

        # 保存结果
        results = {
            'training_metrics': train_metrics,
            'testing_metrics': test_metrics,
            'generator_results': generator_results,
            'model_params': {
                'contamination': self.contamination,
                'n_estimators': self.n_estimators,
                'random_state': self.random_state
            },
            'data_stats': {
                'total_samples': n_samples,
                'train_samples': len(X_train),
                'test_samples': len(X_test),
                'anomaly_ratio': float(np.mean(y)),
                'train_time': train_time,
                'predict_time': predict_time
            }
        }

        with open('isolation_forest_results.json', 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n结果已保存到: isolation_forest_results.json")

        return results

    def _calculate_metrics(self, y_true, y_pred):
        """计算评估指标"""
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        # 混淆矩阵
        cm = confusion_matrix(y_true, y_pred)
        if cm.size == 4:
            tn, fp, fn, tp = cm.ravel()
        else:
            tn, fp, fn, tp = 0, 0, 0, 0

        # 详细指标
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

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
        print(classification_report(y_true, y_pred, target_names=['正常', '异常'], digits=4))

        return {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'specificity': float(specificity),
            'fpr': float(fpr),
            'fnr': float(fnr),
            'confusion_matrix': cm.tolist(),
            'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn)
        }

    def _visualize_results(self, X_test, y_test, y_pred, X_all):
        """可视化结果"""
        # 1. 混淆矩阵
        cm = confusion_matrix(y_test, y_pred)
        self._plot_confusion_matrix(cm, "孤立森林混淆矩阵")

        # 2. 决策边界可视化
        self._plot_decision_boundary(X_test, y_test, y_pred)

        # 3. 异常分数分布
        self._plot_anomaly_scores(X_test, y_test)

    def _plot_confusion_matrix(self, cm, title="混淆矩阵"):
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
        plt.savefig('isolation_forest_confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("混淆矩阵已保存到: isolation_forest_confusion_matrix.png")

    def _plot_decision_boundary(self, X_test, y_true, y_pred):
        """绘制决策边界（仅使用测试集数据）"""
        plt.figure(figsize=(10, 6))

        # 绘制测试集数据点
        x_coords = np.arange(len(X_test))

        # 真实标签
        normal_mask = y_true == 0
        anomaly_mask = y_true == 1

        plt.scatter(x_coords[normal_mask], X_test[normal_mask, 0],
                    c='blue', alpha=0.5, s=10, label='真实正常')
        plt.scatter(x_coords[anomaly_mask], X_test[anomaly_mask, 0],
                    c='red', alpha=0.5, s=10, label='真实异常')

        # 预测错误的点
        correct_mask = y_true == y_pred
        error_mask = y_true != y_pred

        if np.any(error_mask):
            plt.scatter(x_coords[error_mask], X_test[error_mask, 0],
                        c='black', alpha=0.8, s=30, marker='x', label='预测错误')

        plt.xlabel('测试集样本索引', fontsize=12)
        plt.ylabel('标准化值', fontsize=12)
        plt.title('孤立森林检测结果可视化 (测试集)', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('decision_boundary_visualization.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("决策边界可视化已保存到: decision_boundary_visualization.png")

        # 打印预测错误统计
        if np.any(error_mask):
            print(f"预测错误样本数: {np.sum(error_mask)} (占总测试集的 {np.sum(error_mask) / len(y_true) * 100:.2f}%)")

    def _plot_anomaly_scores(self, X_test, y_test):
        """绘制异常分数分布"""
        # 计算异常分数
        anomaly_scores = self.model.decision_function(X_test)

        plt.figure(figsize=(10, 6))

        # 按真实标签分组
        normal_scores = anomaly_scores[y_test == 0]
        anomaly_scores_true = anomaly_scores[y_test == 1]

        # 绘制直方图
        if len(normal_scores) > 0:
            plt.hist(normal_scores, bins=30, alpha=0.7, color='blue',
                     label=f'正常样本 (n={len(normal_scores)})')
        if len(anomaly_scores_true) > 0:
            plt.hist(anomaly_scores_true, bins=30, alpha=0.7, color='red',
                     label=f'异常样本 (n={len(anomaly_scores_true)})')

        # 标记阈值（孤立森林使用0作为阈值）
        plt.axvline(x=0, color='black', linestyle='--', linewidth=2,
                    label='决策边界 (score=0)')

        plt.xlabel('异常分数 (负值表示更异常)', fontsize=12)
        plt.ylabel('频数', fontsize=12)
        plt.title('孤立森林异常分数分布 (测试集)', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('anomaly_scores_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("异常分数分布图已保存到: anomaly_scores_distribution.png")

        # 打印统计信息
        if len(normal_scores) > 0:
            print(f"正常样本平均分数: {np.mean(normal_scores):.4f}")
        if len(anomaly_scores_true) > 0:
            print(f"异常样本平均分数: {np.mean(anomaly_scores_true):.4f}")
        print(f"总体平均分数: {np.mean(anomaly_scores):.4f}")

    def analyze_contamination_effect(self, data_file='dataset.json', contamination_range=None):
        """分析污染参数对模型性能的影响"""
        if contamination_range is None:
            contamination_range = np.arange(0.05, 0.5, 0.05)

        X, y, _ = self.prepare_data(data_file)
        X_scaled = self.scaler.fit_transform(X)

        # 划分训练集和测试集
        n_samples = len(X_scaled)
        split_idx = int(n_samples * 0.7)  # 70%训练，30%测试
        indices = np.random.permutation(n_samples)
        X_train = X_scaled[indices[:split_idx]]
        y_train = y[indices[:split_idx]]
        X_test = X_scaled[indices[split_idx:]]
        y_test = y[indices[split_idx:]]

        f1_scores = []

        print(f"\n{'=' * 60}")
        print("污染参数敏感性分析")
        print(f"{'=' * 60}")

        for contamination in contamination_range:
            # 创建新模型
            model = IsolationForest(
                n_estimators=self.n_estimators,
                contamination=contamination,
                random_state=self.random_state
            )

            # 训练和预测
            model.fit(X_train)
            pred = model.predict(X_test)
            pred_binary = np.where(pred == 1, 0, 1)

            # 计算F1分数
            if len(np.unique(y_test)) > 1 and len(np.unique(pred_binary)) > 1:
                f1 = f1_score(y_test, pred_binary, zero_division=0)
            else:
                f1 = 0.0
            f1_scores.append(f1)

            print(f"污染参数={contamination:.2f}, F1分数={f1:.4f}")

        # 绘制结果
        plt.figure(figsize=(8, 6))
        plt.plot(contamination_range, f1_scores, 'bo-', linewidth=2, markersize=8)
        plt.xlabel('污染参数 (contamination)', fontsize=12)
        plt.ylabel('F1分数 (测试集)', fontsize=12)
        plt.title('污染参数对孤立森林性能的影响', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('contamination_sensitivity.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("污染参数敏感性分析图已保存到: contamination_sensitivity.png")

        # 找到最佳污染参数
        best_idx = np.argmax(f1_scores)
        best_contamination = contamination_range[best_idx]
        best_f1 = f1_scores[best_idx]

        print(f"\n最佳污染参数: {best_contamination:.2f} (F1分数: {best_f1:.4f})")

        return best_contamination, best_f1

    def generate_test_dataset(self):
        """生成测试数据集用于演示"""
        np.random.seed(self.random_state)

        test_data = {}

        # 定义每个发生器的正常范围
        generator_ranges = {
            1: {'min': 0, 'max': 10, 'anomaly_rate': 0.1},
            2: {'min': 1000, 'max': 30000, 'anomaly_rate': 0.15},
            3: {'min': -80, 'max': -20, 'anomaly_rate': 0.2},
            4: {'min': -3.9, 'max': 82.7, 'anomaly_rate': 0.12},
            5: {'min': -1000, 'max': 2000, 'anomaly_rate': 0.18}
        }

        # 为每个发生器生成数据
        for gen_id, params in generator_ranges.items():
            n_samples = 10000  # 每个发生器10000个样本

            # 生成正常数据
            normal_values = np.random.uniform(
                params['min'],
                params['max'],
                int(n_samples * (1 - params['anomaly_rate']))
            )

            # 生成异常数据（超出正常范围）
            anomaly_values = np.concatenate([
                np.random.uniform(params['min'] - 50, params['min'] - 1,
                                  int(n_samples * params['anomaly_rate'] / 2)),
                np.random.uniform(params['max'] + 1, params['max'] + 50,
                                  int(n_samples * params['anomaly_rate'] / 2))
            ])

            # 合并数据
            values = np.concatenate([normal_values, anomaly_values])
            np.random.shuffle(values)

            # 生成标签（0=正常，1=异常）
            labels = np.zeros_like(values)
            labels[-len(anomaly_values):] = 1  # 最后的部分是异常

            # 添加到测试数据
            test_data[f'time_sequence_{gen_id}_value'] = values.tolist()
            test_data[f'time_sequence_{gen_id}_label'] = labels.tolist()

        # 保存测试数据
        with open('test_dataset.json', 'w') as f:
            json.dump(test_data, f)

        print("测试数据集已生成并保存到: test_dataset.json")
        return 'test_dataset.json'


def main():
    """主函数"""
    # 检查数据文件是否存在
    data_file = 'test.json'
    if not os.path.exists(data_file):
        print(f"警告: 数据文件 {data_file} 不存在")
        print("正在生成测试数据集...")
        detector = IsolationForestAnomalyDetector()
        data_file = detector.generate_test_dataset()

    try:
        # 创建孤立森林检测器
        detector = IsolationForestAnomalyDetector(
            contamination=0.1,  # 估计10%的异常
            n_estimators=100,
            random_state=42
        )

        # 分析污染参数影响
        best_contamination, best_f1 = detector.analyze_contamination_effect(data_file)

        # 使用最佳污染参数重新训练
        print(f"\n使用最佳污染参数 {best_contamination:.2f} 重新训练模型...")
        detector = IsolationForestAnomalyDetector(
            contamination=best_contamination,
            n_estimators=100,
            random_state=42
        )

        # 训练和评估模型
        results = detector.train_and_evaluate(data_file, test_size=0.3)

        if results:
            print(f"\n孤立森林检测器评估完成!")
            print(f"测试集F1分数: {results['testing_metrics']['f1_score']:.4f}")

    except Exception as e:
        print(f"程序执行过程中发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    main()

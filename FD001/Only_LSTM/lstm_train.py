import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import json
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import time
import os
from lstm_model import create_model


class TimeSeriesDataset(Dataset):
    """时间序列数据集类"""

    def __init__(self, data_file, sequence_length=10, generator_id=1, normalize=True):
        """
        初始化数据集

        Args:
            data_file: JSON数据文件路径
            sequence_length: 序列长度
            generator_id: 发生器ID (1-5)
            normalize: 是否进行Z-score标准化
        """
        self.sequence_length = sequence_length
        self.generator_id = generator_id
        self.normalize = normalize

        # 加载数据
        with open(data_file, 'r') as f:
            data_dict = json.load(f)

        # 获取指定发生器的数据
        values_key = f"time_sequence_{generator_id}_value"
        labels_key = f"time_sequence_{generator_id}_label"

        self.values = np.array(data_dict[values_key])
        self.labels = np.array(data_dict[labels_key])

        # 数据标准化
        if normalize:
            self.mean = np.mean(self.values)
            self.std = np.std(self.values)
            self.values = (self.values - self.mean) / self.std

        # 创建滑动窗口样本
        self.samples, self.sample_labels = self._create_sequences()

    def _create_sequences(self):
        """创建滑动窗口序列"""
        sequences = []
        labels = []

        for i in range(len(self.values) - self.sequence_length + 1):
            seq = self.values[i:i + self.sequence_length]
            label = self.labels[i + self.sequence_length - 1]  # 窗口最后一个点的标签

            sequences.append(seq)
            labels.append(label)

        return np.array(sequences, dtype=np.float32), np.array(labels, dtype=np.float32)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        label = self.sample_labels[idx]

        # 转换为PyTorch张量
        sample = torch.FloatTensor(sample).unsqueeze(-1)  # 添加特征维度
        label = torch.FloatTensor([label])

        return sample, label


class ModelTrainer:
    """模型训练器"""

    def __init__(self, config=None):
        self.config = config or self.get_default_config()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.optimizer = None
        self.criterion = None

    @staticmethod
    def get_default_config():
        return {
            'data_file': 'train.json',
            'generator_id': 1,
            'sequence_length': 10,
            'batch_size': 64,
            'learning_rate': 0.001,
            'num_epochs': 100,
            'train_ratio': 0.7,
            'val_ratio': 0.15,
            'test_ratio': 0.15,
            'model_config': {
                'input_dim': 1,
                'seq_length': 10,
                'lstm_hidden': 32,
                'lstm_layers': 2,
                'dropout_rate': 0.2
            }
        }

    def setup_data(self):
        """设置数据加载器"""
        dataset = TimeSeriesDataset(
            data_file=self.config['data_file'],
            sequence_length=self.config['sequence_length'],
            generator_id=self.config['generator_id']
        )

        # 数据集划分
        total_size = len(dataset)
        train_size = int(self.config['train_ratio'] * total_size)
        val_size = int(self.config['val_ratio'] * total_size)
        test_size = total_size - train_size - val_size

        train_dataset, val_dataset, test_dataset = random_split(
            dataset, [train_size, val_size, test_size]
        )

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False
        )
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False
        )

        return train_size, val_size, test_size

    def setup_model(self):
        """设置模型、优化器和损失函数"""
        self.model = create_model(self.config['model_config']).to(self.device)
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config['learning_rate']
        )
        self.criterion = nn.BCELoss()

        print(f"模型参数量: {self.model.get_parameter_count():,}")
        print(f"训练设备: {self.device}")

    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        running_loss = 0.0
        all_predictions = []
        all_labels = []

        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

            # 收集预测结果用于计算指标
            predictions = (output > 0.5).float()
            all_predictions.extend(predictions.cpu().detach().numpy())
            all_labels.extend(target.cpu().detach().numpy())

        epoch_loss = running_loss / len(self.train_loader)
        all_predictions = np.array(all_predictions).flatten()
        all_labels = np.array(all_labels).flatten()

        # 计算指标
        precision = precision_score(all_labels, all_predictions, zero_division=0)
        recall = recall_score(all_labels, all_predictions, zero_division=0)
        f1 = f1_score(all_labels, all_predictions, zero_division=0)

        return epoch_loss, precision, recall, f1

    def validate(self, loader):
        """验证或测试"""
        self.model.eval()
        running_loss = 0.0
        all_predictions = []
        all_labels = []
        all_probabilities = []

        with torch.no_grad():
            for data, target in loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)

                running_loss += loss.item()

                probabilities = output.cpu().numpy()
                predictions = (output > 0.5).float()

                all_probabilities.extend(probabilities.flatten())
                all_predictions.extend(predictions.cpu().numpy().flatten())
                all_labels.extend(target.cpu().numpy().flatten())

        loss = running_loss / len(loader)
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        all_probabilities = np.array(all_probabilities)

        # 计算指标
        precision = precision_score(all_labels, all_predictions, zero_division=0)
        recall = recall_score(all_labels, all_predictions, zero_division=0)
        f1 = f1_score(all_labels, all_predictions, zero_division=0)

        # 计算AUC
        if len(np.unique(all_labels)) > 1:
            auc = roc_auc_score(all_labels, all_probabilities)
        else:
            auc = 0.0

        return loss, precision, recall, f1, auc

    def train(self, save_path='pure_lstm_best.pth'):
        """完整训练流程"""
        print("开始设置数据加载器...")
        train_size, val_size, test_size = self.setup_data()
        print(f"数据集大小 - 训练: {train_size}, 验证: {val_size}, 测试: {test_size}")

        print("开始设置模型...")
        self.setup_model()

        best_f1 = 0.0
        training_history = {
            'train_loss': [], 'val_loss': [],
            'train_f1': [], 'val_f1': [],
            'train_precision': [], 'val_precision': [],
            'train_recall': [], 'val_recall': []
        }

        print("开始训练...")
        for epoch in range(self.config['num_epochs']):
            start_time = time.time()

            # 训练
            train_loss, train_precision, train_recall, train_f1 = self.train_epoch(epoch)

            # 验证
            val_loss, val_precision, val_recall, val_f1, val_auc = self.validate(self.val_loader)

            # 记录历史
            training_history['train_loss'].append(train_loss)
            training_history['val_loss'].append(val_loss)
            training_history['train_f1'].append(train_f1)
            training_history['val_f1'].append(val_f1)
            training_history['train_precision'].append(train_precision)
            training_history['val_precision'].append(val_precision)
            training_history['train_recall'].append(train_recall)
            training_history['val_recall'].append(val_recall)

            # 保存最佳模型
            if val_f1 > best_f1:
                best_f1 = val_f1
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_f1': best_f1,
                    'config': self.config
                }, save_path)

            epoch_time = time.time() - start_time

            if epoch % 10 == 0:
                print(f'Epoch {epoch:3d}/{self.config["num_epochs"]} | '
                      f'Time: {epoch_time:.2f}s | '
                      f'Train Loss: {train_loss:.4f} | '
                      f'Val Loss: {val_loss:.4f} | '
                      f'Train F1: {train_f1:.4f} | '
                      f'Val F1: {val_f1:.4f} | '
                      f'Val AUC: {val_auc:.4f}')

        # 加载最佳模型进行测试
        checkpoint = torch.load(save_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        # 测试
        test_loss, test_precision, test_recall, test_f1, test_auc = self.validate(self.test_loader)

        print("\n" + "=" * 50)
        print("最终测试结果:")
        print(f"测试集 F1-Score: {test_f1:.4f}")
        print(f"测试集 Precision: {test_precision:.4f}")
        print(f"测试集 Recall: {test_recall:.4f}")
        print(f"测试集 AUC: {test_auc:.4f}")
        print(f"测试集 Loss: {test_loss:.4f}")
        print("=" * 50)

        return training_history, {
            'test_f1': test_f1, 'test_precision': test_precision,
            'test_recall': test_recall, 'test_auc': test_auc,
            'test_loss': test_loss
        }


def main():
    """主函数"""
    trainer = ModelTrainer()
    history, test_results = trainer.train('pure_lstm_best.pth')

    # 保存训练历史
    import json
    with open('training_history.json', 'w') as f:
        json.dump({
            'history': history,
            'test_results': test_results
        }, f, indent=2)


if __name__ == "__main__":
    main()
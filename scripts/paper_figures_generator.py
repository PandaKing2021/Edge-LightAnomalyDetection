#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
论文图表生成器
基于实验结果为论文生成高质量的图表
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from pathlib import Path
import json
import matplotlib

# 设置中文字体和图表样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

# 创建论文图表目录
paper_figures_dir = Path("results/paper_figures")
paper_figures_dir.mkdir(parents=True, exist_ok=True)

def create_performance_comparison_chart():
    """创建模型性能对比图（柱状图）"""
    print("生成模型性能对比图...")
    
    # 模拟数据集性能数据（从final_experiment_report_updated.md提取）
    models = ['1DCNN-LSTM', '纯LSTM', '纯1D-CNN', '孤立森林', '基于规则']
    f1_scores_simulate = [0.9939, 0.9912, 0.5847, 0.4748, 0.4888]
    accuracy_simulate = [0.9989, 0.9984, 0.9431, 0.9026, 0.9407]
    
    # NASA数据集性能数据
    f1_scores_nasa = [0.3484, 0.3695, 0.3777, 0.3182, 0.0000]
    accuracy_nasa = [0.9163, 0.9236, 0.9265, 0.9481, 0.9746]
    
    # 创建子图
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('模型性能对比分析', fontsize=16, fontweight='bold')
    
    # 1. 模拟数据集F1分数
    x = np.arange(len(models))
    width = 0.35
    
    ax1 = axes[0, 0]
    bars1 = ax1.bar(x, f1_scores_simulate, width, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
    ax1.set_xlabel('模型')
    ax1.set_ylabel('F1分数')
    ax1.set_title('(a) 模拟数据集 - F1分数')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=45, ha='right')
    ax1.set_ylim(0, 1.1)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 在柱子上添加数值标签
    for bar in bars1:
        height = bar.get_height()
        ax1.annotate(f'{height:.4f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 偏移量
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)
    
    # 2. 模拟数据集准确率
    ax2 = axes[0, 1]
    bars2 = ax2.bar(x, accuracy_simulate, width, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
    ax2.set_xlabel('模型')
    ax2.set_ylabel('准确率')
    ax2.set_title('(b) 模拟数据集 - 准确率')
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, rotation=45, ha='right')
    ax2.set_ylim(0.8, 1.1)
    ax2.grid(True, alpha=0.3, axis='y')
    
    for bar in bars2:
        height = bar.get_height()
        ax2.annotate(f'{height:.4f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)
    
    # 3. NASA数据集F1分数
    ax3 = axes[1, 0]
    bars3 = ax3.bar(x, f1_scores_nasa, width, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
    ax3.set_xlabel('模型')
    ax3.set_ylabel('F1分数')
    ax3.set_title('(c) NASA FD001数据集 - F1分数')
    ax3.set_xticks(x)
    ax3.set_xticklabels(models, rotation=45, ha='right')
    ax3.set_ylim(0, 0.5)
    ax3.grid(True, alpha=0.3, axis='y')
    
    for bar in bars3:
        height = bar.get_height()
        ax3.annotate(f'{height:.4f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)
    
    # 4. NASA数据集准确率
    ax4 = axes[1, 1]
    bars4 = ax4.bar(x, accuracy_nasa, width, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
    ax4.set_xlabel('模型')
    ax4.set_ylabel('准确率')
    ax4.set_title('(d) NASA FD001数据集 - 准确率')
    ax4.set_xticks(x)
    ax4.set_xticklabels(models, rotation=45, ha='right')
    ax4.set_ylim(0.8, 1.0)
    ax4.grid(True, alpha=0.3, axis='y')
    
    for bar in bars4:
        height = bar.get_height()
        ax4.annotate(f'{height:.4f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    fig_path = paper_figures_dir / "performance_comparison.png"
    plt.savefig(fig_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"  保存至: {fig_path}")
    
    # 生成简化版性能对比图（仅F1分数）
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 模拟数据集F1分数
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    bars_sim = ax1.bar(models, f1_scores_simulate, color=colors)
    ax1.set_title('模拟数据集 - F1分数对比', fontsize=14, fontweight='bold')
    ax1.set_ylabel('F1分数')
    ax1.set_ylim(0, 1.1)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.tick_params(axis='x', rotation=45)
    
    for bar, val in zip(bars_sim, f1_scores_simulate):
        height = bar.get_height()
        ax1.annotate(f'{val:.4f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)
    
    # NASA数据集F1分数
    bars_nasa = ax2.bar(models, f1_scores_nasa, color=colors)
    ax2.set_title('NASA FD001数据集 - F1分数对比', fontsize=14, fontweight='bold')
    ax2.set_ylabel('F1分数')
    ax2.set_ylim(0, 0.5)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.tick_params(axis='x', rotation=45)
    
    for bar, val in zip(bars_nasa, f1_scores_nasa):
        height = bar.get_height()
        ax2.annotate(f'{val:.4f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    fig_path2 = paper_figures_dir / "f1_comparison_simplified.png"
    plt.savefig(fig_path2, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"  保存简化版至: {fig_path2}")

def create_ablation_analysis_chart():
    """创建消融实验分析图"""
    print("生成消融实验分析图...")
    
    # 消融实验数据（模拟数据集）
    models_ablation = ['1DCNN-LSTM混合模型', '纯LSTM', '纯1D-CNN']
    metrics = ['F1分数', '准确率', '召回率', '精确率']
    
    # 性能数据
    f1_scores = [0.9939, 0.9912, 0.5847]
    accuracy = [0.9989, 0.9984, 0.9431]
    recall = [0.9913, 0.9863, 0.4362]
    precision = [0.9965, 0.9963, 0.8866]
    
    # 创建雷达图
    fig = plt.figure(figsize=(10, 8))
    
    # 计算雷达图角度
    num_vars = len(metrics)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # 闭合图形
    
    # 准备数据
    values_hybrid = [f1_scores[0], accuracy[0], recall[0], precision[0]]
    values_lstm = [f1_scores[1], accuracy[1], recall[1], precision[1]]
    values_cnn = [f1_scores[2], accuracy[2], recall[2], precision[2]]
    
    # 闭合数据
    values_hybrid += values_hybrid[:1]
    values_lstm += values_lstm[:1]
    values_cnn += values_cnn[:1]
    
    # 创建雷达图
    ax = fig.add_subplot(111, polar=True)
    ax.plot(angles, values_hybrid, 'o-', linewidth=2, label='1DCNN-LSTM混合模型', color='#1f77b4')
    ax.fill(angles, values_hybrid, alpha=0.25, color='#1f77b4')
    ax.plot(angles, values_lstm, 'o-', linewidth=2, label='纯LSTM', color='#ff7f0e')
    ax.fill(angles, values_lstm, alpha=0.25, color='#ff7f0e')
    ax.plot(angles, values_cnn, 'o-', linewidth=2, label='纯1D-CNN', color='#2ca02c')
    ax.fill(angles, values_cnn, alpha=0.25, color='#2ca02c')
    
    # 设置标签
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=12)
    ax.set_ylim(0, 1.1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=10)
    ax.set_title('消融实验分析 - 模拟数据集', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    ax.grid(True)
    
    plt.tight_layout()
    fig_path = paper_figures_dir / "ablation_analysis_radar.png"
    plt.savefig(fig_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"  保存至: {fig_path}")
    
    # 创建柱状图对比
    fig2, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(models_ablation))
    width = 0.2
    
    bars1 = ax.bar(x - 1.5*width, f1_scores, width, label='F1分数', color='#1f77b4')
    bars2 = ax.bar(x - 0.5*width, accuracy, width, label='准确率', color='#ff7f0e')
    bars3 = ax.bar(x + 0.5*width, recall, width, label='召回率', color='#2ca02c')
    bars4 = ax.bar(x + 1.5*width, precision, width, label='精确率', color='#d62728')
    
    ax.set_xlabel('模型')
    ax.set_ylabel('性能指标值')
    ax.set_title('消融实验 - 各模型性能指标对比', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models_ablation)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # 添加数值标签
    def autolabel(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.4f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)
    
    autolabel(bars1)
    autolabel(bars2)
    autolabel(bars3)
    autolabel(bars4)
    
    plt.tight_layout()
    fig_path2 = paper_figures_dir / "ablation_analysis_bar.png"
    plt.savefig(fig_path2, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"  保存柱状图至: {fig_path2}")

def create_quantization_effect_chart():
    """创建量化效果对比图"""
    print("生成量化效果对比图...")
    
    # 量化效果数据（模拟数据集）
    metrics = ['准确率', 'F1分数', '推理时间(ms)', '加速比']
    original = [0.9564, 0.6761, 500.00, 1.00]
    quantized = [0.9576, 0.6887, 385.00, 1.30]
    
    # 创建对比图
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('INT8量化效果对比 - 模拟数据集', fontsize=16, fontweight='bold')
    
    colors = ['#1f77b4', '#ff7f0e']
    labels = ['原始模型(FP32)', '量化模型(INT8)']
    
    # 1. 准确率对比
    ax1 = axes[0, 0]
    x_pos = [0, 1]
    bars1 = ax1.bar(x_pos, [original[0], quantized[0]], color=colors)
    ax1.set_ylabel('准确率')
    ax1.set_title('(a) 准确率对比')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(labels)
    ax1.set_ylim(0.95, 0.96)
    ax1.grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars1, [original[0], quantized[0]]):
        height = bar.get_height()
        ax1.annotate(f'{val:.4f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)
    
    # 2. F1分数对比
    ax2 = axes[0, 1]
    bars2 = ax2.bar(x_pos, [original[1], quantized[1]], color=colors)
    ax2.set_ylabel('F1分数')
    ax2.set_title('(b) F1分数对比')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(labels)
    ax2.set_ylim(0.67, 0.70)
    ax2.grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars2, [original[1], quantized[1]]):
        height = bar.get_height()
        ax2.annotate(f'{val:.4f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)
    
    # 3. 推理时间对比
    ax3 = axes[1, 0]
    bars3 = ax3.bar(x_pos, [original[2], quantized[2]], color=colors)
    ax3.set_ylabel('推理时间(ms)')
    ax3.set_title('(c) 推理时间对比')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(labels)
    ax3.set_ylim(0, 550)
    ax3.grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars3, [original[2], quantized[2]]):
        height = bar.get_height()
        ax3.annotate(f'{val:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)
    
    # 4. 加速比对比
    ax4 = axes[1, 1]
    bars4 = ax4.bar(x_pos, [original[3], quantized[3]], color=colors)
    ax4.set_ylabel('加速比')
    ax4.set_title('(d) 加速比对比')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(labels)
    ax4.set_ylim(0, 1.5)
    ax4.grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars4, [original[3], quantized[3]]):
        height = bar.get_height()
        ax4.annotate(f'{val:.2f}x',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    fig_path = paper_figures_dir / "quantization_effect_comparison.png"
    plt.savefig(fig_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"  保存至: {fig_path}")
    
    # 创建量化效果综合图
    fig2, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(metrics))
    width = 0.35
    
    bars_orig = ax.bar(x - width/2, original, width, label='原始模型(FP32)', color='#1f77b4')
    bars_quant = ax.bar(x + width/2, quantized, width, label='量化模型(INT8)', color='#ff7f0e')
    
    ax.set_xlabel('指标')
    ax.set_ylabel('值')
    ax.set_title('INT8量化效果综合对比', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # 添加变化百分比
    for i, (orig, quant) in enumerate(zip(original, quantized)):
        if i < 2:  # 准确率和F1分数
            change = (quant - orig) / orig * 100
            ax.text(i, max(orig, quant) + 0.02, f'+{change:.2f}%', 
                   ha='center', fontsize=9)
        elif i == 2:  # 推理时间
            reduction = (orig - quant) / orig * 100
            ax.text(i, max(orig, quant) + 20, f'-{reduction:.1f}%', 
                   ha='center', fontsize=9)
    
    plt.tight_layout()
    fig_path2 = paper_figures_dir / "quantization_effect_summary.png"
    plt.savefig(fig_path2, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"  保存综合图至: {fig_path2}")

def create_cooperative_framework_chart():
    """创建协同推理框架性能图"""
    print("生成协同推理框架性能图...")
    
    # 从final_cooperative_optimization_report.md提取数据
    thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
    f1_scores = [0.9946, 0.9956, 0.9943, 0.9919, 0.9944]
    energy_saving = [7.47, 65.24, 69.88, 71.03, 81.96]
    wakeup_rate = [0.4878, 0.1758, 0.1507, 0.1446, 0.0855]
    
    # 创建双Y轴图
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # F1分数（左轴）
    color1 = '#1f77b4'
    ax1.set_xlabel('决策阈值')
    ax1.set_ylabel('F1分数', color=color1)
    line1 = ax1.plot(thresholds, f1_scores, 'o-', color=color1, linewidth=2, markersize=8, label='F1分数')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_ylim(0.98, 1.0)
    ax1.grid(True, alpha=0.3)
    
    # 在点上添加数值标签
    for i, (x, y) in enumerate(zip(thresholds, f1_scores)):
        ax1.annotate(f'{y:.4f}', xy=(x, y), xytext=(0, 10),
                    textcoords='offset points', ha='center', va='bottom',
                    fontsize=9, color=color1)
    
    # 能耗节省率（右轴）
    ax2 = ax1.twinx()
    color2 = '#ff7f0e'
    ax2.set_ylabel('能耗节省率 (%)', color=color2)
    line2 = ax2.plot(thresholds, energy_saving, 's-', color=color2, linewidth=2, markersize=8, label='能耗节省率')
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.set_ylim(0, 100)
    
    # 在点上添加数值标签
    for i, (x, y) in enumerate(zip(thresholds, energy_saving)):
        ax2.annotate(f'{y:.1f}%', xy=(x, y), xytext=(0, -15),
                    textcoords='offset points', ha='center', va='top',
                    fontsize=9, color=color2)
    
    # 标记最佳平衡点（阈值=0.9）
    best_idx = 4  # 阈值=0.9
    ax1.axvline(x=thresholds[best_idx], color='red', linestyle='--', alpha=0.7, linewidth=1.5)
    ax1.text(thresholds[best_idx], 0.985, f'最佳平衡点\n阈值={thresholds[best_idx]}', 
            ha='center', va='bottom', fontsize=10, color='red',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    # 合并图例
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper right')
    
    ax1.set_title('端-边协同推理框架性能分析\n（LSTM作为哨兵模型）', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    fig_path = paper_figures_dir / "cooperative_framework_performance.png"
    plt.savefig(fig_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"  保存至: {fig_path}")
    
    # 创建唤醒率图
    fig2, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(thresholds, wakeup_rate, 'o-', color='#2ca02c', linewidth=2, markersize=8)
    ax.set_xlabel('决策阈值')
    ax.set_ylabel('唤醒率')
    ax.set_title('端-边协同推理框架 - 唤醒率分析', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 0.6)
    
    # 添加数值标签
    for i, (x, y) in enumerate(zip(thresholds, wakeup_rate)):
        ax.annotate(f'{y:.4f}', xy=(x, y), xytext=(0, 10),
                   textcoords='offset points', ha='center', va='bottom',
                   fontsize=10)
    
    # 标记最佳平衡点
    ax.axvline(x=thresholds[best_idx], color='red', linestyle='--', alpha=0.7, linewidth=1.5)
    ax.text(thresholds[best_idx], wakeup_rate[best_idx]/2, f'阈值={thresholds[best_idx]}\n唤醒率={wakeup_rate[best_idx]:.4f}', 
           ha='center', va='center', fontsize=10, color='red',
           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    fig_path2 = paper_figures_dir / "cooperative_framework_wakeup_rate.png"
    plt.savefig(fig_path2, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"  保存唤醒率图至: {fig_path2}")
    
    # 创建3D图（阈值、F1、能耗节省）
    from mpl_toolkits.mplot3d import Axes3D
    
    fig3 = plt.figure(figsize=(12, 8))
    ax3d = fig3.add_subplot(111, projection='3d')
    
    # 生成更多数据点以平滑曲线
    thresholds_dense = np.linspace(0.1, 0.9, 20)
    # 使用插值生成平滑曲线
    from scipy.interpolate import interp1d
    f1_interp = interp1d(thresholds, f1_scores, kind='cubic')
    energy_interp = interp1d(thresholds, energy_saving, kind='cubic')
    wakeup_interp = interp1d(thresholds, wakeup_rate, kind='cubic')
    
    f1_dense = f1_interp(thresholds_dense)
    energy_dense = energy_interp(thresholds_dense)
    
    # 3D散点图
    scatter = ax3d.scatter(thresholds, f1_scores, energy_saving, 
                          c=wakeup_rate, cmap='viridis', s=100, alpha=0.8)
    
    # 3D曲线
    ax3d.plot(thresholds_dense, f1_dense, energy_dense, 'r-', alpha=0.6, linewidth=2)
    
    ax3d.set_xlabel('决策阈值', fontsize=12)
    ax3d.set_ylabel('F1分数', fontsize=12)
    ax3d.set_zlabel('能耗节省率 (%)', fontsize=12)
    ax3d.set_title('协同推理框架性能三维分析', fontsize=14, fontweight='bold')
    
    # 添加颜色条
    cbar = fig3.colorbar(scatter, ax=ax3d, pad=0.1)
    cbar.set_label('唤醒率', rotation=270, labelpad=15)
    
    # 标记最佳点
    ax3d.scatter([thresholds[best_idx]], [f1_scores[best_idx]], [energy_saving[best_idx]], 
                c='red', s=200, marker='*', label='最佳平衡点')
    ax3d.text(thresholds[best_idx], f1_scores[best_idx], energy_saving[best_idx],
             f' 最佳点\n F1={f1_scores[best_idx]:.4f}\n 节能={energy_saving[best_idx]:.1f}%',
             fontsize=9, color='red')
    
    ax3d.legend()
    
    plt.tight_layout()
    fig_path3 = paper_figures_dir / "cooperative_framework_3d.png"
    plt.savefig(fig_path3, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"  保存3D图至: {fig_path3}")

def create_optimization_comparison_chart():
    """创建优化前后对比图"""
    print("生成优化前后对比图...")
    
    # 优化前后数据
    scenarios = ['原始方案(1DCNN哨兵)', '优化方案(LSTM哨兵)']
    
    # 阈值=0.9时的性能
    f1_scores = [0.0836, 0.9944]  # 从报告提取
    energy_saving = [35.54, 81.96]
    
    # 最佳F1分数（不同阈值）
    best_f1 = [0.4378, 0.9944]  # 原始方案在阈值=0.1，优化方案在阈值=0.9
    
    # 创建对比图
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('协同推理框架优化效果对比', fontsize=16, fontweight='bold')
    
    colors = ['#d62728', '#2ca02c']
    
    # 1. 阈值=0.9时F1分数对比
    ax1 = axes[0]
    bars1 = ax1.bar(scenarios, f1_scores, color=colors)
    ax1.set_ylabel('F1分数')
    ax1.set_title('(a) 阈值=0.9时F1分数对比')
    ax1.set_ylim(0, 1.1)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.tick_params(axis='x', rotation=15)
    
    for bar, val in zip(bars1, f1_scores):
        height = bar.get_height()
        ax1.annotate(f'{val:.4f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)
    
    # 计算改进百分比
    improvement_f1 = (f1_scores[1] - f1_scores[0]) / f1_scores[0] * 100
    ax1.text(0.5, f1_scores[1] + 0.05, f'提升: +{improvement_f1:.1f}%', 
            ha='center', fontsize=10, fontweight='bold', color='green')
    
    # 2. 能耗节省率对比
    ax2 = axes[1]
    bars2 = ax2.bar(scenarios, energy_saving, color=colors)
    ax2.set_ylabel('能耗节省率 (%)')
    ax2.set_title('(b) 阈值=0.9时能耗节省率对比')
    ax2.set_ylim(0, 100)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.tick_params(axis='x', rotation=15)
    
    for bar, val in zip(bars2, energy_saving):
        height = bar.get_height()
        ax2.annotate(f'{val:.2f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)
    
    # 计算改进百分点
    improvement_energy = energy_saving[1] - energy_saving[0]
    ax2.text(0.5, energy_saving[1] + 5, f'提升: +{improvement_energy:.1f}个百分点', 
            ha='center', fontsize=10, fontweight='bold', color='green')
    
    # 3. 最佳F1分数对比
    ax3 = axes[2]
    bars3 = ax3.bar(scenarios, best_f1, color=colors)
    ax3.set_ylabel('最佳F1分数')
    ax3.set_title('(c) 最佳F1分数对比')
    ax3.set_ylim(0, 1.1)
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.tick_params(axis='x', rotation=15)
    
    for bar, val in zip(bars3, best_f1):
        height = bar.get_height()
        ax3.annotate(f'{val:.4f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)
    
    # 计算改进百分比
    improvement_best_f1 = (best_f1[1] - best_f1[0]) / best_f1[0] * 100
    ax3.text(0.5, best_f1[1] + 0.05, f'提升: +{improvement_best_f1:.1f}%', 
            ha='center', fontsize=10, fontweight='bold', color='green')
    
    plt.tight_layout()
    fig_path = paper_figures_dir / "optimization_comparison.png"
    plt.savefig(fig_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"  保存至: {fig_path}")
    
    # 创建综合改进图
    fig2, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(scenarios))
    width = 0.25
    
    # 标准化数据到0-1范围以便对比
    f1_norm = [f1 / 1.0 for f1 in f1_scores]  # F1最大为1
    energy_norm = [e / 100.0 for e in energy_saving]  # 能耗节省率最大为100%
    best_f1_norm = [bf / 1.0 for bf in best_f1]
    
    bars1 = ax.bar(x - width, f1_norm, width, label='F1分数(阈值=0.9)', color='#1f77b4')
    bars2 = ax.bar(x, energy_norm, width, label='能耗节省率', color='#ff7f0e')
    bars3 = ax.bar(x + width, best_f1_norm, width, label='最佳F1分数', color='#2ca02c')
    
    ax.set_xlabel('方案')
    ax.set_ylabel('标准化值')
    ax.set_title('协同推理框架优化效果综合对比', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # 添加实际值标签
    def add_value_labels(bars, values, offset=0.02):
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.annotate(f'{val:.4f}' if val < 1 else f'{val:.1f}%',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)
    
    add_value_labels(bars1, f1_scores)
    add_value_labels(bars2, energy_saving)
    add_value_labels(bars3, best_f1)
    
    plt.tight_layout()
    fig_path2 = paper_figures_dir / "optimization_comparison_summary.png"
    plt.savefig(fig_path2, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"  保存综合图至: {fig_path2}")

def create_summary_figure():
    """创建论文摘要图（一图概括所有关键结果）"""
    print("生成论文摘要图...")
    
    fig = plt.figure(figsize=(16, 12))
    
    # 定义布局
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. 性能对比（左上）
    ax1 = fig.add_subplot(gs[0, 0])
    models = ['1DCNN-LSTM', '纯LSTM', '纯1D-CNN']
    f1_scores = [0.9939, 0.9912, 0.5847]
    bars = ax1.bar(models, f1_scores, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    ax1.set_title('(a) 模型性能对比（模拟数据集）', fontsize=11)
    ax1.set_ylabel('F1分数')
    ax1.set_ylim(0, 1.1)
    ax1.tick_params(axis='x', rotation=45)
    for bar, val in zip(bars, f1_scores):
        height = bar.get_height()
        ax1.annotate(f'{val:.4f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords='offset points', ha='center', va='bottom', fontsize=9)
    
    # 2. 消融分析（中上）
    ax2 = fig.add_subplot(gs[0, 1])
    metrics = ['F1', 'Acc', 'Rec', 'Prec']
    hybrid = [0.9939, 0.9989, 0.9913, 0.9965]
    lstm = [0.9912, 0.9984, 0.9863, 0.9963]
    cnn = [0.5847, 0.9431, 0.4362, 0.8866]
    
    x = np.arange(len(metrics))
    width = 0.25
    ax2.bar(x - width, hybrid, width, label='混合模型', color='#1f77b4')
    ax2.bar(x, lstm, width, label='纯LSTM', color='#ff7f0e')
    ax2.bar(x + width, cnn, width, label='纯1D-CNN', color='#2ca02c')
    ax2.set_title('(b) 消融实验分析', fontsize=11)
    ax2.set_ylabel('性能值')
    ax2.set_xticks(x)
    ax2.set_xticklabels(metrics)
    ax2.legend(fontsize=9)
    ax2.set_ylim(0, 1.1)
    
    # 3. 量化效果（右上）
    ax3 = fig.add_subplot(gs[0, 2])
    metrics_q = ['准确率', 'F1分数', '推理时间\n(降低)', '加速比']
    original_q = [0.9564, 0.6761, 500, 1.0]
    quantized_q = [0.9576, 0.6887, 385, 1.3]
    
    x_q = np.arange(len(metrics_q))
    ax3.bar(x_q - 0.2, original_q, 0.4, label='原始(FP32)', color='#1f77b4')
    ax3.bar(x_q + 0.2, quantized_q, 0.4, label='量化(INT8)', color='#ff7f0e')
    ax3.set_title('(c) INT8量化效果', fontsize=11)
    ax3.set_ylabel('值')
    ax3.set_xticks(x_q)
    ax3.set_xticklabels(metrics_q, fontsize=9)
    ax3.legend(fontsize=9)
    
    # 4. 协同框架性能（中左）
    ax4 = fig.add_subplot(gs[1, :2])
    thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
    f1_coop = [0.9946, 0.9956, 0.9943, 0.9919, 0.9944]
    energy = [7.47, 65.24, 69.88, 71.03, 81.96]
    
    ax4.plot(thresholds, f1_coop, 'o-', color='#1f77b4', label='F1分数', linewidth=2)
    ax4.set_xlabel('决策阈值')
    ax4.set_ylabel('F1分数', color='#1f77b4')
    ax4.tick_params(axis='y', labelcolor='#1f77b4')
    ax4.set_ylim(0.98, 1.0)
    ax4.set_title('(d) 协同推理框架性能分析', fontsize=11)
    
    ax4_2 = ax4.twinx()
    ax4_2.plot(thresholds, energy, 's-', color='#ff7f0e', label='能耗节省率', linewidth=2)
    ax4_2.set_ylabel('能耗节省率 (%)', color='#ff7f0e')
    ax4_2.tick_params(axis='y', labelcolor='#ff7f0e')
    ax4_2.set_ylim(0, 100)
    
    # 标记最佳点
    best_idx = 4
    ax4.axvline(x=thresholds[best_idx], color='red', linestyle='--', alpha=0.7)
    ax4.text(thresholds[best_idx], 0.985, f'最佳点\nF1={f1_coop[best_idx]:.4f}\n节能={energy[best_idx]:.1f}%',
            ha='center', va='bottom', fontsize=9, color='red')
    
    # 合并图例
    lines1, labels1 = ax4.get_legend_handles_labels()
    lines2, labels2 = ax4_2.get_legend_handles_labels()
    ax4.legend(lines1 + lines2, labels1 + labels2, loc='lower right', fontsize=9)
    
    # 5. 优化前后对比（中右）
    ax5 = fig.add_subplot(gs[1, 2])
    scenarios = ['原始方案', '优化方案']
    f1_comp = [0.0836, 0.9944]
    energy_comp = [35.54, 81.96]
    
    x_comp = np.arange(len(scenarios))
    width_comp = 0.35
    bars5_1 = ax5.bar(x_comp - width_comp/2, f1_comp, width_comp, label='F1分数', color='#1f77b4')
    bars5_2 = ax5.bar(x_comp + width_comp/2, energy_comp, width_comp, label='能耗节省率', color='#ff7f0e')
    ax5.set_title('(e) 优化效果对比（阈值=0.9）', fontsize=11)
    ax5.set_xticks(x_comp)
    ax5.set_xticklabels(scenarios)
    ax5.set_ylabel('值')
    ax5.legend(fontsize=9)
    
    # 添加改进标签
    ax5.text(0.5, max(f1_comp[1], energy_comp[1]) * 0.9, 
            f'F1: +{(f1_comp[1]-f1_comp[0])/f1_comp[0]*100:.0f}%\n节能: +{energy_comp[1]-energy_comp[0]:.1f}%',
            ha='center', fontsize=9, fontweight='bold', color='green')
    
    # 6. 关键指标总结（底部）
    ax6 = fig.add_subplot(gs[2, :])
    ax6.axis('off')
    
    summary_text = """
    论文核心实验结果总结
    
    1. 模型性能: 1DCNN-LSTM混合模型在模拟数据集上取得最佳F1分数 (0.9939)，验证了架构有效性
    2. 消融分析: 混合模型相比纯LSTM提升0.27%，相比纯1D-CNN提升70.0%，证明模块协同效应显著
    3. 量化效果: INT8量化实现1.30×推理加速，精度损失小于0.2%，适合边缘部署
    4. 协同框架: 端-边协同推理框架在阈值=0.9时实现F1=0.9944且能耗节省81.96%，完全满足设计要求
    5. 优化效果: 通过哨兵模型优化（LSTM替代1D-CNN），F1分数提升127.1%，能耗节省率提升46.4个百分点
    
    结论: 本文提出的轻量级1DCNN-LSTM混合模型及端-边协同推理框架，在保证检测性能的前提下
          显著降低系统能耗，为边缘时序异常检测提供了有效的解决方案。
    """
    
    ax6.text(0.05, 0.95, summary_text, fontsize=11, va='top',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))
    
    plt.suptitle('论文实验结果摘要图', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    fig_path = paper_figures_dir / "paper_summary_figure.png"
    plt.savefig(fig_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"  保存至: {fig_path}")

def main():
    """主函数"""
    print("=" * 60)
    print("论文图表生成器")
    print("=" * 60)
    
    # 创建所有图表
    create_performance_comparison_chart()
    create_ablation_analysis_chart()
    create_quantization_effect_chart()
    create_cooperative_framework_chart()
    create_optimization_comparison_chart()
    create_summary_figure()
    
    print("\n" + "=" * 60)
    print("所有图表生成完成!")
    print(f"图表保存在: {paper_figures_dir}")
    print("=" * 60)
    
    # 生成图表清单
    chart_list = """
    生成的论文图表清单:
    
    1. 性能对比图:
       - performance_comparison.png: 综合性能对比图（4个子图）
       - f1_comparison_simplified.png: F1分数简化对比图
    
    2. 消融实验图:
       - ablation_analysis_radar.png: 消融实验雷达图
       - ablation_analysis_bar.png: 消融实验柱状图
    
    3. 量化效果图:
       - quantization_effect_comparison.png: 量化效果对比图（4个子图）
       - quantization_effect_summary.png: 量化效果综合图
    
    4. 协同推理框架图:
       - cooperative_framework_performance.png: 框架性能分析图（双Y轴）
       - cooperative_framework_wakeup_rate.png: 唤醒率分析图
       - cooperative_framework_3d.png: 三维性能分析图
    
    5. 优化效果图:
       - optimization_comparison.png: 优化前后对比图
       - optimization_comparison_summary.png: 优化效果综合图
    
    6. 论文摘要图:
       - paper_summary_figure.png: 一图概括所有关键结果
    
    所有图表均为300 DPI，适合论文出版使用。
    """
    
    # 保存清单到文件
    list_path = paper_figures_dir / "chart_list.md"
    with open(list_path, 'w', encoding='utf-8') as f:
        f.write(chart_list)
    
    print(chart_list)
    print(f"图表清单保存至: {list_path}")

if __name__ == "__main__":
    main()
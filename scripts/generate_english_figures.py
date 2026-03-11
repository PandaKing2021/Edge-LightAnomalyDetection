#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
English Figures Generator
Generate English versions of all paper figures based on existing Chinese figures.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from pathlib import Path
import json
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import interp1d

# Set English font and chart style
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Helvetica']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

# Create English figures directory
english_figures_dir = Path("results/english_figures")
english_figures_dir.mkdir(parents=True, exist_ok=True)

def create_performance_comparison_chart():
    """Create model performance comparison chart (bar chart)"""
    print("Generating model performance comparison chart...")
    
    # Simulated dataset performance data (extracted from final_experiment_report_updated.md)
    models = ['1DCNN-LSTM', 'Pure LSTM', 'Pure 1D-CNN', 'Isolation Forest', 'Rule-based']
    f1_scores_simulate = [0.9939, 0.9912, 0.5847, 0.4748, 0.4888]
    accuracy_simulate = [0.9989, 0.9984, 0.9431, 0.9026, 0.9407]
    
    # NASA dataset performance data
    f1_scores_nasa = [0.3484, 0.3695, 0.3777, 0.3182, 0.0000]
    accuracy_nasa = [0.9163, 0.9236, 0.9265, 0.9481, 0.9746]
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Model Performance Comparison Analysis', fontsize=16, fontweight='bold')
    
    # 1. Simulated dataset F1 scores
    x = np.arange(len(models))
    width = 0.35
    
    ax1 = axes[0, 0]
    bars1 = ax1.bar(x, f1_scores_simulate, width, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
    ax1.set_xlabel('Model')
    ax1.set_ylabel('F1 Score')
    ax1.set_title('(a) Simulated Dataset - F1 Score')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=45, ha='right')
    ax1.set_ylim(0, 1.1)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.annotate(f'{height:.4f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)
    
    # 2. Simulated dataset accuracy
    ax2 = axes[0, 1]
    bars2 = ax2.bar(x, accuracy_simulate, width, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
    ax2.set_xlabel('Model')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('(b) Simulated Dataset - Accuracy')
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
    
    # 3. NASA dataset F1 scores
    ax3 = axes[1, 0]
    bars3 = ax3.bar(x, f1_scores_nasa, width, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
    ax3.set_xlabel('Model')
    ax3.set_ylabel('F1 Score')
    ax3.set_title('(c) NASA FD001 Dataset - F1 Score')
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
    
    # 4. NASA dataset accuracy
    ax4 = axes[1, 1]
    bars4 = ax4.bar(x, accuracy_nasa, width, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
    ax4.set_xlabel('Model')
    ax4.set_ylabel('Accuracy')
    ax4.set_title('(d) NASA FD001 Dataset - Accuracy')
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
    fig_path = english_figures_dir / "performance_comparison.png"
    plt.savefig(fig_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"  Saved to: {fig_path}")
    
    # Generate simplified performance comparison chart (F1 scores only)
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Simulated dataset F1 scores
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    bars_sim = ax1.bar(models, f1_scores_simulate, color=colors)
    ax1.set_title('Simulated Dataset - F1 Score Comparison', fontsize=14, fontweight='bold')
    ax1.set_ylabel('F1 Score')
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
    
    # NASA dataset F1 scores
    bars_nasa = ax2.bar(models, f1_scores_nasa, color=colors)
    ax2.set_title('NASA FD001 Dataset - F1 Score Comparison', fontsize=14, fontweight='bold')
    ax2.set_ylabel('F1 Score')
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
    fig_path2 = english_figures_dir / "f1_comparison_simplified.png"
    plt.savefig(fig_path2, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"  Saved simplified version to: {fig_path2}")

def create_ablation_analysis_chart():
    """Create ablation experiment analysis chart"""
    print("Generating ablation experiment analysis chart...")
    
    # Ablation experiment data (simulated dataset)
    models_ablation = ['1DCNN-LSTM Hybrid Model', 'Pure LSTM', 'Pure 1D-CNN']
    metrics = ['F1 Score', 'Accuracy', 'Recall', 'Precision']
    
    # Performance data
    f1_scores = [0.9939, 0.9912, 0.5847]
    accuracy = [0.9989, 0.9984, 0.9431]
    recall = [0.9913, 0.9863, 0.4362]
    precision = [0.9965, 0.9963, 0.8866]
    
    # Create radar chart
    fig = plt.figure(figsize=(10, 8))
    
    # Calculate radar chart angles
    num_vars = len(metrics)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # Close the shape
    
    # Prepare data
    values_hybrid = [f1_scores[0], accuracy[0], recall[0], precision[0]]
    values_lstm = [f1_scores[1], accuracy[1], recall[1], precision[1]]
    values_cnn = [f1_scores[2], accuracy[2], recall[2], precision[2]]
    
    # Close data
    values_hybrid += values_hybrid[:1]
    values_lstm += values_lstm[:1]
    values_cnn += values_cnn[:1]
    
    # Create radar chart
    ax = fig.add_subplot(111, polar=True)
    ax.plot(angles, values_hybrid, 'o-', linewidth=2, label='1DCNN-LSTM Hybrid Model', color='#1f77b4')
    ax.fill(angles, values_hybrid, alpha=0.25, color='#1f77b4')
    ax.plot(angles, values_lstm, 'o-', linewidth=2, label='Pure LSTM', color='#ff7f0e')
    ax.fill(angles, values_lstm, alpha=0.25, color='#ff7f0e')
    ax.plot(angles, values_cnn, 'o-', linewidth=2, label='Pure 1D-CNN', color='#2ca02c')
    ax.fill(angles, values_cnn, alpha=0.25, color='#2ca02c')
    
    # Set labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=12)
    ax.set_ylim(0, 1.1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=10)
    ax.set_title('Ablation Experiment Analysis - Simulated Dataset', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    ax.grid(True)
    
    plt.tight_layout()
    fig_path = english_figures_dir / "ablation_analysis_radar.png"
    plt.savefig(fig_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"  Saved to: {fig_path}")
    
    # Create bar chart comparison
    fig2, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(models_ablation))
    width = 0.2
    
    bars1 = ax.bar(x - 1.5*width, f1_scores, width, label='F1 Score', color='#1f77b4')
    bars2 = ax.bar(x - 0.5*width, accuracy, width, label='Accuracy', color='#ff7f0e')
    bars3 = ax.bar(x + 0.5*width, recall, width, label='Recall', color='#2ca02c')
    bars4 = ax.bar(x + 1.5*width, precision, width, label='Precision', color='#d62728')
    
    ax.set_xlabel('Model')
    ax.set_ylabel('Performance Metric Value')
    ax.set_title('Ablation Experiment - Performance Metrics Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models_ablation)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
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
    fig_path2 = english_figures_dir / "ablation_analysis_bar.png"
    plt.savefig(fig_path2, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"  Saved bar chart to: {fig_path2}")

def create_quantization_effect_chart():
    """Create quantization effect comparison chart"""
    print("Generating quantization effect comparison chart...")
    
    # Quantization effect data (simulated dataset)
    metrics = ['Accuracy', 'F1 Score', 'Inference Time(ms)', 'Speedup Ratio']
    original = [0.9564, 0.6761, 500.00, 1.00]
    quantized = [0.9576, 0.6887, 385.00, 1.30]
    
    # Create comparison chart
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('INT8 Quantization Effect Comparison - Simulated Dataset', fontsize=16, fontweight='bold')
    
    colors = ['#1f77b4', '#ff7f0e']
    labels = ['Original Model(FP32)', 'Quantized Model(INT8)']
    
    # 1. Accuracy comparison
    ax1 = axes[0, 0]
    x_pos = [0, 1]
    bars1 = ax1.bar(x_pos, [original[0], quantized[0]], color=colors)
    ax1.set_ylabel('Accuracy')
    ax1.set_title('(a) Accuracy Comparison')
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
    
    # 2. F1 score comparison
    ax2 = axes[0, 1]
    bars2 = ax2.bar(x_pos, [original[1], quantized[1]], color=colors)
    ax2.set_ylabel('F1 Score')
    ax2.set_title('(b) F1 Score Comparison')
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
    
    # 3. Inference time comparison
    ax3 = axes[1, 0]
    bars3 = ax3.bar(x_pos, [original[2], quantized[2]], color=colors)
    ax3.set_ylabel('Inference Time(ms)')
    ax3.set_title('(c) Inference Time Comparison')
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
    
    # 4. Speedup ratio comparison
    ax4 = axes[1, 1]
    bars4 = ax4.bar(x_pos, [original[3], quantized[3]], color=colors)
    ax4.set_ylabel('Speedup Ratio')
    ax4.set_title('(d) Speedup Ratio Comparison')
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
    fig_path = english_figures_dir / "quantization_effect_comparison.png"
    plt.savefig(fig_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"  Saved to: {fig_path}")
    
    # Create quantization effect summary chart
    fig2, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(metrics))
    width = 0.35
    
    bars_orig = ax.bar(x - width/2, original, width, label='Original Model(FP32)', color='#1f77b4')
    bars_quant = ax.bar(x + width/2, quantized, width, label='Quantized Model(INT8)', color='#ff7f0e')
    
    ax.set_xlabel('Metric')
    ax.set_ylabel('Value')
    ax.set_title('INT8 Quantization Effect Summary', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add change percentage
    for i, (orig, quant) in enumerate(zip(original, quantized)):
        if i < 2:  # Accuracy and F1 score
            change = (quant - orig) / orig * 100
            ax.text(i, max(orig, quant) + 0.02, f'+{change:.2f}%', 
                   ha='center', fontsize=9)
        elif i == 2:  # Inference time
            reduction = (orig - quant) / orig * 100
            ax.text(i, max(orig, quant) + 20, f'-{reduction:.1f}%', 
                   ha='center', fontsize=9)
    
    plt.tight_layout()
    fig_path2 = english_figures_dir / "quantization_effect_summary.png"
    plt.savefig(fig_path2, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"  Saved summary chart to: {fig_path2}")

def create_cooperative_framework_chart():
    """Create cooperative inference framework performance chart"""
    print("Generating cooperative inference framework performance chart...")
    
    # Data extracted from final_cooperative_optimization_report.md
    thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
    f1_scores = [0.9946, 0.9956, 0.9943, 0.9919, 0.9944]
    energy_saving = [7.47, 65.24, 69.88, 71.03, 81.96]
    wakeup_rate = [0.4878, 0.1758, 0.1507, 0.1446, 0.0855]
    
    # Create dual Y-axis chart
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # F1 score (left axis)
    color1 = '#1f77b4'
    ax1.set_xlabel('Decision Threshold')
    ax1.set_ylabel('F1 Score', color=color1)
    line1 = ax1.plot(thresholds, f1_scores, 'o-', color=color1, linewidth=2, markersize=8, label='F1 Score')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_ylim(0.98, 1.0)
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on points
    for i, (x, y) in enumerate(zip(thresholds, f1_scores)):
        ax1.annotate(f'{y:.4f}', xy=(x, y), xytext=(0, 10),
                    textcoords='offset points', ha='center', va='bottom',
                    fontsize=9, color=color1)
    
    # Energy saving rate (right axis)
    ax2 = ax1.twinx()
    color2 = '#ff7f0e'
    ax2.set_ylabel('Energy Saving Rate (%)', color=color2)
    line2 = ax2.plot(thresholds, energy_saving, 's-', color=color2, linewidth=2, markersize=8, label='Energy Saving Rate')
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.set_ylim(0, 100)
    
    # Add value labels on points
    for i, (x, y) in enumerate(zip(thresholds, energy_saving)):
        ax2.annotate(f'{y:.1f}%', xy=(x, y), xytext=(0, -15),
                    textcoords='offset points', ha='center', va='top',
                    fontsize=9, color=color2)
    
    # Mark best balance point (threshold=0.9)
    best_idx = 4  # threshold=0.9
    ax1.axvline(x=thresholds[best_idx], color='red', linestyle='--', alpha=0.7, linewidth=1.5)
    ax1.text(thresholds[best_idx], 0.985, f'Best Balance Point\nThreshold={thresholds[best_idx]}', 
            ha='center', va='bottom', fontsize=10, color='red',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper right')
    
    ax1.set_title('Edge-Cloud Cooperative Inference Framework Performance Analysis\n(LSTM as Sentinel Model)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    fig_path = english_figures_dir / "cooperative_framework_performance.png"
    plt.savefig(fig_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"  Saved to: {fig_path}")
    
    # Create wakeup rate chart
    fig2, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(thresholds, wakeup_rate, 'o-', color='#2ca02c', linewidth=2, markersize=8)
    ax.set_xlabel('Decision Threshold')
    ax.set_ylabel('Wakeup Rate')
    ax.set_title('Edge-Cloud Cooperative Inference Framework - Wakeup Rate Analysis', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 0.6)
    
    # Add value labels
    for i, (x, y) in enumerate(zip(thresholds, wakeup_rate)):
        ax.annotate(f'{y:.4f}', xy=(x, y), xytext=(0, 10),
                   textcoords='offset points', ha='center', va='bottom',
                   fontsize=10)
    
    # Mark best balance point
    ax.axvline(x=thresholds[best_idx], color='red', linestyle='--', alpha=0.7, linewidth=1.5)
    ax.text(thresholds[best_idx], wakeup_rate[best_idx]/2, f'Threshold={thresholds[best_idx]}\nWakeup Rate={wakeup_rate[best_idx]:.4f}', 
           ha='center', va='center', fontsize=10, color='red',
           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    fig_path2 = english_figures_dir / "cooperative_framework_wakeup_rate.png"
    plt.savefig(fig_path2, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"  Saved wakeup rate chart to: {fig_path2}")
    
    # Create 3D chart (threshold, F1, energy saving)
    fig3 = plt.figure(figsize=(12, 8))
    ax3d = fig3.add_subplot(111, projection='3d')
    
    # Generate more data points for smooth curves
    thresholds_dense = np.linspace(0.1, 0.9, 20)
    # Use interpolation for smooth curves
    f1_interp = interp1d(thresholds, f1_scores, kind='cubic')
    energy_interp = interp1d(thresholds, energy_saving, kind='cubic')
    wakeup_interp = interp1d(thresholds, wakeup_rate, kind='cubic')
    
    f1_dense = f1_interp(thresholds_dense)
    energy_dense = energy_interp(thresholds_dense)
    
    # 3D scatter plot
    scatter = ax3d.scatter(thresholds, f1_scores, energy_saving, 
                          c=wakeup_rate, cmap='viridis', s=100, alpha=0.8)
    
    # 3D curve
    ax3d.plot(thresholds_dense, f1_dense, energy_dense, 'r-', alpha=0.6, linewidth=2)
    
    ax3d.set_xlabel('Decision Threshold', fontsize=12)
    ax3d.set_ylabel('F1 Score', fontsize=12)
    ax3d.set_zlabel('Energy Saving Rate (%)', fontsize=12)
    ax3d.set_title('Cooperative Inference Framework 3D Performance Analysis', fontsize=14, fontweight='bold')
    
    # Add color bar
    cbar = fig3.colorbar(scatter, ax=ax3d, pad=0.1)
    cbar.set_label('Wakeup Rate', rotation=270, labelpad=15)
    
    # Mark best point
    ax3d.scatter([thresholds[best_idx]], [f1_scores[best_idx]], [energy_saving[best_idx]], 
                c='red', s=200, marker='*', label='Best Balance Point')
    ax3d.text(thresholds[best_idx], f1_scores[best_idx], energy_saving[best_idx],
             f'  Best Point\n  F1={f1_scores[best_idx]:.4f}\n  Energy Saving={energy_saving[best_idx]:.1f}%',
             fontsize=9, color='red')
    
    ax3d.legend()
    
    plt.tight_layout()
    fig_path3 = english_figures_dir / "cooperative_framework_3d.png"
    plt.savefig(fig_path3, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"  Saved 3D chart to: {fig_path3}")

def create_optimization_comparison_chart():
    """Create optimization comparison chart"""
    print("Generating optimization comparison chart...")
    
    # Before and after optimization data
    scenarios = ['Original Scheme(1DCNN Sentinel)', 'Optimized Scheme(LSTM Sentinel)']
    
    # Performance at threshold=0.9
    f1_scores = [0.0836, 0.9944]  # Extracted from report
    energy_saving = [35.54, 81.96]
    
    # Best F1 score (different thresholds)
    best_f1 = [0.4378, 0.9944]  # Original scheme at threshold=0.1, optimized scheme at threshold=0.9
    
    # Create comparison chart
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Cooperative Inference Framework Optimization Effect Comparison', fontsize=16, fontweight='bold')
    
    colors = ['#d62728', '#2ca02c']
    
    # 1. F1 score comparison at threshold=0.9
    ax1 = axes[0]
    bars1 = ax1.bar(scenarios, f1_scores, color=colors)
    ax1.set_ylabel('F1 Score')
    ax1.set_title('(a) F1 Score Comparison (Threshold=0.9)')
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
    
    # Calculate improvement percentage
    improvement_f1 = (f1_scores[1] - f1_scores[0]) / f1_scores[0] * 100
    ax1.text(0.5, f1_scores[1] + 0.05, f'Improvement: +{improvement_f1:.1f}%', 
            ha='center', fontsize=10, fontweight='bold', color='green')
    
    # 2. Energy saving rate comparison
    ax2 = axes[1]
    bars2 = ax2.bar(scenarios, energy_saving, color=colors)
    ax2.set_ylabel('Energy Saving Rate (%)')
    ax2.set_title('(b) Energy Saving Rate Comparison (Threshold=0.9)')
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
    
    # Calculate improvement in percentage points
    improvement_energy = energy_saving[1] - energy_saving[0]
    ax2.text(0.5, energy_saving[1] + 5, f'Improvement: +{improvement_energy:.1f} percentage points', 
            ha='center', fontsize=10, fontweight='bold', color='green')
    
    # 3. Best F1 score comparison
    ax3 = axes[2]
    bars3 = ax3.bar(scenarios, best_f1, color=colors)
    ax3.set_ylabel('Best F1 Score')
    ax3.set_title('(c) Best F1 Score Comparison')
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
    
    # Calculate improvement percentage
    improvement_best_f1 = (best_f1[1] - best_f1[0]) / best_f1[0] * 100
    ax3.text(0.5, best_f1[1] + 0.05, f'Improvement: +{improvement_best_f1:.1f}%', 
            ha='center', fontsize=10, fontweight='bold', color='green')
    
    plt.tight_layout()
    fig_path = english_figures_dir / "optimization_comparison.png"
    plt.savefig(fig_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"  Saved to: {fig_path}")
    
    # Create comprehensive improvement chart
    fig2, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(scenarios))
    width = 0.25
    
    # Normalize data to 0-1 range for comparison
    f1_norm = [f1 / 1.0 for f1 in f1_scores]  # F1 max is 1
    energy_norm = [e / 100.0 for e in energy_saving]  # Energy saving rate max is 100%
    best_f1_norm = [bf / 1.0 for bf in best_f1]
    
    bars1 = ax.bar(x - width, f1_norm, width, label='F1 Score(Threshold=0.9)', color='#1f77b4')
    bars2 = ax.bar(x, energy_norm, width, label='Energy Saving Rate', color='#ff7f0e')
    bars3 = ax.bar(x + width, best_f1_norm, width, label='Best F1 Score', color='#2ca02c')
    
    ax.set_xlabel('Scheme')
    ax.set_ylabel('Normalized Value')
    ax.set_title('Cooperative Inference Framework Optimization Effect Comprehensive Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add actual value labels
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
    fig_path2 = english_figures_dir / "optimization_comparison_summary.png"
    plt.savefig(fig_path2, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"  Saved summary chart to: {fig_path2}")

def create_summary_figure():
    """Create paper summary figure (one figure summarizing all key results)"""
    print("Generating paper summary figure...")
    
    fig = plt.figure(figsize=(16, 12))
    
    # Define layout
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Performance comparison (top left)
    ax1 = fig.add_subplot(gs[0, 0])
    models = ['1DCNN-LSTM', 'Pure LSTM', 'Pure 1D-CNN']
    f1_scores = [0.9939, 0.9912, 0.5847]
    bars = ax1.bar(models, f1_scores, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    ax1.set_title('(a) Model Performance Comparison (Simulated Dataset)', fontsize=11)
    ax1.set_ylabel('F1 Score')
    ax1.set_ylim(0, 1.1)
    ax1.tick_params(axis='x', rotation=45)
    for bar, val in zip(bars, f1_scores):
        height = bar.get_height()
        ax1.annotate(f'{val:.4f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords='offset points', ha='center', va='bottom', fontsize=9)
    
    # 2. Ablation analysis (top middle)
    ax2 = fig.add_subplot(gs[0, 1])
    metrics = ['F1', 'Acc', 'Rec', 'Prec']
    hybrid = [0.9939, 0.9989, 0.9913, 0.9965]
    lstm = [0.9912, 0.9984, 0.9863, 0.9963]
    cnn = [0.5847, 0.9431, 0.4362, 0.8866]
    
    x = np.arange(len(metrics))
    width = 0.25
    ax2.bar(x - width, hybrid, width, label='Hybrid Model', color='#1f77b4')
    ax2.bar(x, lstm, width, label='Pure LSTM', color='#ff7f0e')
    ax2.bar(x + width, cnn, width, label='Pure 1D-CNN', color='#2ca02c')
    ax2.set_title('(b) Ablation Experiment Analysis', fontsize=11)
    ax2.set_ylabel('Performance Value')
    ax2.set_xticks(x)
    ax2.set_xticklabels(metrics)
    ax2.legend(fontsize=9)
    ax2.set_ylim(0, 1.1)
    
    # 3. Quantization effect (top right)
    ax3 = fig.add_subplot(gs[0, 2])
    metrics_q = ['Accuracy', 'F1 Score', 'Inference Time\n(Reduction)', 'Speedup Ratio']
    original_q = [0.9564, 0.6761, 500, 1.0]
    quantized_q = [0.9576, 0.6887, 385, 1.3]
    
    x_q = np.arange(len(metrics_q))
    ax3.bar(x_q - 0.2, original_q, 0.4, label='Original(FP32)', color='#1f77b4')
    ax3.bar(x_q + 0.2, quantized_q, 0.4, label='Quantized(INT8)', color='#ff7f0e')
    ax3.set_title('(c) INT8 Quantization Effect', fontsize=11)
    ax3.set_ylabel('Value')
    ax3.set_xticks(x_q)
    ax3.set_xticklabels(metrics_q, fontsize=9)
    ax3.legend(fontsize=9)
    
    # 4. Cooperative framework performance (middle left)
    ax4 = fig.add_subplot(gs[1, :2])
    thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
    f1_coop = [0.9946, 0.9956, 0.9943, 0.9919, 0.9944]
    energy = [7.47, 65.24, 69.88, 71.03, 81.96]
    
    ax4.plot(thresholds, f1_coop, 'o-', color='#1f77b4', label='F1 Score', linewidth=2)
    ax4.set_xlabel('Decision Threshold')
    ax4.set_ylabel('F1 Score', color='#1f77b4')
    ax4.tick_params(axis='y', labelcolor='#1f77b4')
    ax4.set_ylim(0.98, 1.0)
    ax4.set_title('(d) Cooperative Inference Framework Performance Analysis', fontsize=11)
    
    ax4_2 = ax4.twinx()
    ax4_2.plot(thresholds, energy, 's-', color='#ff7f0e', label='Energy Saving Rate', linewidth=2)
    ax4_2.set_ylabel('Energy Saving Rate (%)', color='#ff7f0e')
    ax4_2.tick_params(axis='y', labelcolor='#ff7f0e')
    ax4_2.set_ylim(0, 100)
    
    # Mark best point
    best_idx = 4
    ax4.axvline(x=thresholds[best_idx], color='red', linestyle='--', alpha=0.7)
    ax4.text(thresholds[best_idx], 0.985, f'Best Point\nF1={f1_coop[best_idx]:.4f}\nEnergy Saving={energy[best_idx]:.1f}%',
            ha='center', va='bottom', fontsize=9, color='red')
    
    # Combine legends
    lines1, labels1 = ax4.get_legend_handles_labels()
    lines2, labels2 = ax4_2.get_legend_handles_labels()
    ax4.legend(lines1 + lines2, labels1 + labels2, loc='lower right', fontsize=9)
    
    # 5. Optimization comparison (middle right)
    ax5 = fig.add_subplot(gs[1, 2])
    scenarios = ['Original Scheme', 'Optimized Scheme']
    f1_comp = [0.0836, 0.9944]
    energy_comp = [35.54, 81.96]
    
    x_comp = np.arange(len(scenarios))
    width_comp = 0.35
    bars5_1 = ax5.bar(x_comp - width_comp/2, f1_comp, width_comp, label='F1 Score', color='#1f77b4')
    bars5_2 = ax5.bar(x_comp + width_comp/2, energy_comp, width_comp, label='Energy Saving Rate', color='#ff7f0e')
    ax5.set_title('(e) Optimization Effect Comparison (Threshold=0.9)', fontsize=11)
    ax5.set_xticks(x_comp)
    ax5.set_xticklabels(scenarios)
    ax5.set_ylabel('Value')
    ax5.legend(fontsize=9)
    
    # Add improvement labels
    ax5.text(0.5, max(f1_comp[1], energy_comp[1]) * 0.9, 
            f'F1: +{(f1_comp[1]-f1_comp[0])/f1_comp[0]*100:.0f}%\nEnergy: +{energy_comp[1]-energy_comp[0]:.1f}%',
            ha='center', fontsize=9, fontweight='bold', color='green')
    
    # 6. Key metrics summary (bottom)
    ax6 = fig.add_subplot(gs[2, :])
    ax6.axis('off')
    
    summary_text = """
    Summary of Core Experimental Results
    
    1. Model Performance: 1DCNN-LSTM hybrid model achieves the best F1 score (0.9939) on simulated dataset, validating architecture effectiveness
    2. Ablation Analysis: Hybrid model improves by 0.27% compared to pure LSTM and 70.0% compared to pure 1D-CNN, demonstrating significant module synergy
    3. Quantization Effect: INT8 quantization achieves 1.30× inference acceleration with less than 0.2% accuracy loss, suitable for edge deployment
    4. Cooperative Framework: Edge-cloud cooperative inference framework achieves F1=0.9944 and energy saving of 81.96% at threshold=0.9, fully meeting design requirements
    5. Optimization Effect: Through sentinel model optimization (LSTM replacing 1D-CNN), F1 score improves by 127.1% and energy saving rate improves by 46.4 percentage points
    
    Conclusion: The lightweight 1DCNN-LSTM hybrid model and edge-cloud cooperative inference framework proposed in this paper
          significantly reduce system energy consumption while ensuring detection performance, providing an effective solution
          for edge time-series anomaly detection.
    """
    
    ax6.text(0.05, 0.95, summary_text, fontsize=11, va='top',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))
    
    plt.suptitle('Paper Experimental Results Summary Figure', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    fig_path = english_figures_dir / "paper_summary_figure.png"
    plt.savefig(fig_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"  Saved to: {fig_path}")

def create_final_optimization_charts():
    """Create English versions of final optimization charts"""
    print("Generating final optimization charts...")
    
    # Load CSV data
    csv_path = Path("results/final_optimization/optimized_system_results.csv")
    if not csv_path.exists():
        print(f"  Warning: CSV file not found at {csv_path}")
        return
    
    df = pd.read_csv(csv_path)
    
    # 1. 3D Threshold Impact Chart
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    scatter = ax.scatter(df['threshold'], df['f1_score'], df['energy_saving'], 
                        c=df['wakeup_rate'], cmap='viridis', s=80, alpha=0.8)
    
    ax.set_xlabel('Decision Threshold', fontsize=12)
    ax.set_ylabel('F1 Score', fontsize=12)
    ax.set_zlabel('Energy Saving Rate (%)', fontsize=12)
    ax.set_title('3D Threshold Impact Analysis (LSTM Sentinel)', fontsize=14, fontweight='bold')
    
    cbar = fig.colorbar(scatter, ax=ax, pad=0.1)
    cbar.set_label('Wakeup Rate', rotation=270, labelpad=15)
    
    # Mark optimal point (threshold=0.9)
    optimal_idx = df[df['threshold'] == 0.9].index[0]
    ax.scatter([df.loc[optimal_idx, 'threshold']], 
               [df.loc[optimal_idx, 'f1_score']], 
               [df.loc[optimal_idx, 'energy_saving']], 
               c='red', s=200, marker='*', label='Optimal Point (Threshold=0.9)')
    
    ax.legend()
    plt.tight_layout()
    fig_path = english_figures_dir / "3d_threshold_impact.png"
    plt.savefig(fig_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"  Saved 3D threshold impact chart to: {fig_path}")
    
    # 2. Performance-Efficiency Tradeoff Chart
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    color1 = '#1f77b4'
    ax1.set_xlabel('Decision Threshold')
    ax1.set_ylabel('F1 Score', color=color1)
    ax1.plot(df['threshold'], df['f1_score'], 'o-', color=color1, linewidth=2, markersize=6, label='F1 Score')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_ylim(0.99, 1.0)
    ax1.grid(True, alpha=0.3)
    
    ax2 = ax1.twinx()
    color2 = '#ff7f0e'
    ax2.set_ylabel('Energy Saving Rate (%)', color=color2)
    ax2.plot(df['threshold'], df['energy_saving'], 's-', color=color2, linewidth=2, markersize=6, label='Energy Saving Rate')
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.set_ylim(0, 100)
    
    # Mark optimal point
    ax1.axvline(x=0.9, color='red', linestyle='--', alpha=0.7, linewidth=1.5)
    ax1.text(0.9, 0.992, f'Optimal Point\nF1={df.loc[optimal_idx, "f1_score"]:.4f}\nEnergy Saving={df.loc[optimal_idx, "energy_saving"]:.1f}%',
            ha='center', va='bottom', fontsize=10, color='red',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    ax1.set_title('Performance-Efficiency Tradeoff Analysis (LSTM Sentinel)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig_path = english_figures_dir / "final_performance_efficiency_tradeoff.png"
    plt.savefig(fig_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"  Saved performance-efficiency tradeoff chart to: {fig_path}")
    
    # 3. Model Comparison Chart (Original vs Optimized)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Original scheme data (from report)
    original_f1 = 0.0836  # at threshold=0.9
    original_energy = 35.54
    optimized_f1 = df.loc[optimal_idx, 'f1_score']
    optimized_energy = df.loc[optimal_idx, 'energy_saving']
    
    # F1 comparison
    ax1 = axes[0]
    models = ['Original Scheme\n(1DCNN Sentinel)', 'Optimized Scheme\n(LSTM Sentinel)']
    f1_values = [original_f1, optimized_f1]
    colors = ['#d62728', '#2ca02c']
    bars1 = ax1.bar(models, f1_values, color=colors)
    ax1.set_ylabel('F1 Score')
    ax1.set_title('Model Comparison - F1 Score (Threshold=0.9)')
    ax1.set_ylim(0, 1.1)
    ax1.grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars1, f1_values):
        height = bar.get_height()
        ax1.annotate(f'{val:.4f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords='offset points', ha='center', va='bottom', fontsize=10)
    
    improvement_f1 = (optimized_f1 - original_f1) / original_f1 * 100
    ax1.text(0.5, max(f1_values) + 0.05, f'Improvement: +{improvement_f1:.1f}%',
            ha='center', fontsize=10, fontweight='bold', color='green')
    
    # Energy saving comparison
    ax2 = axes[1]
    energy_values = [original_energy, optimized_energy]
    bars2 = ax2.bar(models, energy_values, color=colors)
    ax2.set_ylabel('Energy Saving Rate (%)')
    ax2.set_title('Model Comparison - Energy Saving (Threshold=0.9)')
    ax2.set_ylim(0, 100)
    ax2.grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars2, energy_values):
        height = bar.get_height()
        ax2.annotate(f'{val:.1f}%', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords='offset points', ha='center', va='bottom', fontsize=10)
    
    improvement_energy = optimized_energy - original_energy
    ax2.text(0.5, max(energy_values) + 5, f'Improvement: +{improvement_energy:.1f} percentage points',
            ha='center', fontsize=10, fontweight='bold', color='green')
    
    plt.tight_layout()
    fig_path = english_figures_dir / "model_comparison_chart.png"
    plt.savefig(fig_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"  Saved model comparison chart to: {fig_path}")
    
    # 4. System Metrics vs Threshold Chart
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))
    
    # F1 score vs threshold
    ax1 = axes[0]
    ax1.plot(df['threshold'], df['f1_score'], 'o-', color='#1f77b4', linewidth=2, markersize=6)
    ax1.set_xlabel('Decision Threshold')
    ax1.set_ylabel('F1 Score')
    ax1.set_title('F1 Score vs Decision Threshold')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0.99, 1.0)
    
    # Energy saving vs threshold
    ax2 = axes[1]
    ax2.plot(df['threshold'], df['energy_saving'], 's-', color='#ff7f0e', linewidth=2, markersize=6)
    ax2.set_xlabel('Decision Threshold')
    ax2.set_ylabel('Energy Saving Rate (%)')
    ax2.set_title('Energy Saving Rate vs Decision Threshold')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 100)
    
    # Wakeup rate vs threshold
    ax3 = axes[2]
    ax3.plot(df['threshold'], df['wakeup_rate'], '^-', color='#2ca02c', linewidth=2, markersize=6)
    ax3.set_xlabel('Decision Threshold')
    ax3.set_ylabel('Wakeup Rate')
    ax3.set_title('Wakeup Rate vs Decision Threshold')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 0.6)
    
    # Mark optimal point on all subplots
    for ax in axes:
        ax.axvline(x=0.9, color='red', linestyle='--', alpha=0.7, linewidth=1)
    
    plt.tight_layout()
    fig_path = english_figures_dir / "system_metrics_vs_threshold.png"
    plt.savefig(fig_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"  Saved system metrics vs threshold chart to: {fig_path}")

def main():
    """Main function"""
    print("=" * 60)
    print("English Figures Generator")
    print("=" * 60)
    
    # Create all charts
    create_performance_comparison_chart()
    create_ablation_analysis_chart()
    create_quantization_effect_chart()
    create_cooperative_framework_chart()
    create_optimization_comparison_chart()
    create_summary_figure()
    create_final_optimization_charts()
    
    print("\n" + "=" * 60)
    print("All English charts generated!")
    print(f"Charts saved in: {english_figures_dir}")
    print("=" * 60)
    
    # Generate chart list
    chart_list = """
    Generated English Paper Charts List:
    
    1. Performance Comparison Charts:
       - performance_comparison.png: Comprehensive performance comparison chart (4 subplots)
       - f1_comparison_simplified.png: Simplified F1 score comparison chart
    
    2. Ablation Experiment Charts:
       - ablation_analysis_radar.png: Ablation experiment radar chart
       - ablation_analysis_bar.png: Ablation experiment bar chart
    
    3. Quantization Effect Charts:
       - quantization_effect_comparison.png: Quantization effect comparison chart (4 subplots)
       - quantization_effect_summary.png: Quantization effect summary chart
    
    4. Cooperative Inference Framework Charts:
       - cooperative_framework_performance.png: Framework performance analysis chart (dual Y-axis)
       - cooperative_framework_wakeup_rate.png: Wakeup rate analysis chart
       - cooperative_framework_3d.png: 3D performance analysis chart
    
    5. Optimization Effect Charts:
       - optimization_comparison.png: Optimization comparison chart
       - optimization_comparison_summary.png: Optimization effect summary chart
    
    6. Paper Summary Figure:
       - paper_summary_figure.png: One figure summarizing all key results
    
    7. Final Optimization Charts:
       - 3d_threshold_impact.png: 3D threshold impact analysis chart
       - final_performance_efficiency_tradeoff.png: Performance-efficiency tradeoff analysis chart
       - model_comparison_chart.png: Model comparison chart (original vs optimized)
       - system_metrics_vs_threshold.png: System metrics vs threshold chart
    
    All charts are 300 DPI, suitable for paper publication.
    """
    
    # Save list to file
    list_path = english_figures_dir / "chart_list.md"
    with open(list_path, 'w', encoding='utf-8') as f:
        f.write(chart_list)
    
    print(chart_list)
    print(f"Chart list saved to: {list_path}")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
论文图片生成脚本
生成论文中最重要的8个对比分析图表
"""

import os
import sys
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from pathlib import Path
from matplotlib import font_manager
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# 配置参数（放在前面，方便调整）
# =============================================================================

# 路径配置
BASE_DIR = Path(__file__).parent.parent
RESULTS_DIR = BASE_DIR / '02-Processing' / 'results'
OUTPUT_DIR = BASE_DIR / '03-Comparison' / 'figures'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 数据文件路径
CSV_FILE = RESULTS_DIR / 'experiment_summary.csv'  # 数据来源：实验汇总表
PKL_DIR = RESULTS_DIR  # 数据来源：实验结果PKL文件目录

# 字体配置
CHINESE_FONT = 'SimSun'  # 宋体
ENGLISH_FONT = 'Times New Roman'  # Times字体
FONT_SIZE_TITLE = 16  # 标题字体大小
FONT_SIZE_LABEL = 14  # 坐标轴标签字体大小
FONT_SIZE_TICK = 12  # 刻度字体大小
FONT_SIZE_LEGEND = 12  # 图例字体大小
FONT_SIZE_SUPTITLE = 18  # 总标题字体大小

# 图片配置
DPI = 300  # 图片分辨率
FIG_SIZE = (10, 6)  # 默认图片大小（宽，高）
FIG_SIZE_WIDE = (14, 6)  # 宽图
FIG_SIZE_TALL = (8, 10)  # 高图

# 颜色配置
COLORS = {
    'ARIMA': '#1f77b4',      # 蓝色
    'LSTM': '#ff7f0e',       # 橙色
    '1D-CNN': '#2ca02c',     # 绿色
    'Transformer': '#d62728' # 红色
}

MODEL_NAMES = ['ARIMA', 'LSTM', '1D-CNN', 'Transformer']
MODEL_NAMES_CN = {
    'ARIMA': 'ARIMA',
    'LSTM': 'LSTM',
    '1D-CNN': '1D-CNN',
    'Transformer': 'Transformer'
}

# 序列长度和预测长度配置
SEQ_LENGTHS = [6, 60, 120, 360]
PRED_LENGTHS = [1, 6, 120, 360]
SEQ_LABELS = ['6步\n(0.5分钟)', '60步\n(5分钟)', '120步\n(10分钟)', '360步\n(30分钟)']
PRED_LABELS = ['1步\n(5秒)', '6步\n(0.5分钟)', '120步\n(10分钟)', '360步\n(30分钟)']

# PKL文件配置（用于预测效果图）
# 注意：特征索引将从PKL文件中的feature_mapping自动获取，如果PKL文件中没有，则使用以下默认值
PKL_CONFIG = {
    'seq_len': 60,  # 代表性序列长度
    'pred_len': 1,  # 代表性预测长度
    'feature_indices': {
        '贯入度': 0,  # 默认值，实际会从PKL文件的feature_mapping中读取
        '推进压力': 1,  # 默认值，实际会从PKL文件的feature_mapping中读取
        '刀盘转速': 20  # 默认值，实际会从PKL文件的feature_mapping中读取（注意：应该是20，不是18）
    },
    'sample_range': (0, 500)  # 展示的样本范围（从测试集中选择前500个样本）
}

# =============================================================================
# 字体设置
# =============================================================================

def setup_fonts():
    """设置中英文字体"""
    plt.rcParams['font.sans-serif'] = [CHINESE_FONT, 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['font.serif'] = [ENGLISH_FONT, 'Times', 'DejaVu Serif']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['figure.dpi'] = DPI
    plt.rcParams['savefig.dpi'] = DPI
    plt.rcParams['savefig.bbox'] = 'tight'
    sns.set_style("whitegrid")
    sns.set_palette("husl")

def set_ticklabels_bold(ax):
    """为坐标轴刻度标签添加加粗"""
    for label in ax.get_xticklabels():
        label.set_fontweight('bold')
    for label in ax.get_yticklabels():
        label.set_fontweight('bold')

# =============================================================================
# 数据加载
# =============================================================================

def load_data():
    """
    加载实验数据
    数据来源：Transformer/02-Processing/results/experiment_summary.csv
    包含64个实验组合的结果（4种模型 × 4种序列长度 × 4种预测长度）
    """
    if not CSV_FILE.exists():
        raise FileNotFoundError(f"数据文件不存在: {CSV_FILE}")
    
    df = pd.read_csv(CSV_FILE)
    print(f"已加载数据: {len(df)} 条实验记录")
    return df

def load_pkl_data(model_name, seq_len, pred_len):
    """
    加载PKL文件数据
    数据来源：Transformer/02-Processing/results/{model_name}_{seq_len}_{pred_len}_results.pkl
    包含：y_true, y_pred, y_true_inv, y_pred_inv等预测结果数据
    """
    pkl_file = PKL_DIR / f'{model_name}_{seq_len}_{pred_len}_results.pkl'
    if not pkl_file.exists():
        return None
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)
    return data

# =============================================================================
# 图片生成函数（8个最重要的图片）
# =============================================================================

def figure1_model_accuracy_comparison(df):
    """
    图1：四种模型预测精度对比
    数据来源：experiment_summary.csv
    - R²值：df.groupby('model')['R2'].mean()
    - MSE、MAE、RMSE：df.groupby('model')[metric].mean()
    展示4个子图：R²、MSE、MAE、RMSE
    """
    fig, axes = plt.subplots(4, 1, figsize=(10, 14))
    
    metrics = ['R2', 'MSE', 'MAE', 'RMSE']
    metric_names = ['R²', 'MSE', 'MAE', 'RMSE']
    metric_titles = ['R²值', 'MSE', 'MAE', 'RMSE']
    
    for idx, (metric, name, title) in enumerate(zip(metrics, metric_names, metric_titles)):
        ax = axes[idx]
        
        if metric == 'R2':
            # R²值越大越好，降序排列
            model_metric = df.groupby('model')[metric].mean().sort_values(ascending=False)
        else:
            # 误差指标越小越好，升序排列
            model_metric = df.groupby('model')[metric].mean().sort_values(ascending=True)
        
        # 统一使用垂直柱状图
        bars = ax.bar(range(len(model_metric)), model_metric.values,
                     color=[COLORS[m] for m in model_metric.index])
        
        # 添加数值标签
        for i, (bar, val) in enumerate(zip(bars, model_metric.values)):
            if metric == 'R2':
                ax.text(bar.get_x() + bar.get_width()/2, val + 0.01,
                       f'{val:.3f}', ha='center', va='bottom',
                       fontsize=FONT_SIZE_TICK, fontfamily=ENGLISH_FONT, fontweight='bold')
            else:
                ax.text(bar.get_x() + bar.get_width()/2, val + val*0.02,
                       f'{val:.4f}', ha='center', va='bottom',
                       fontsize=FONT_SIZE_TICK, fontfamily=ENGLISH_FONT, fontweight='bold')
        
        ax.set_xticks(range(len(model_metric)))
        ax.set_xticklabels([MODEL_NAMES_CN[m] for m in model_metric.index],
                          fontsize=FONT_SIZE_LABEL, fontfamily=CHINESE_FONT, fontweight='bold')
        ax.set_ylabel(title, fontsize=FONT_SIZE_LABEL, fontfamily=CHINESE_FONT, fontweight='bold')
        ax.set_title(f'({chr(97+idx)}) {title}', fontsize=FONT_SIZE_TITLE,
                    fontfamily=CHINESE_FONT, fontweight='bold', loc='left')
        set_ticklabels_bold(ax)
        ax.grid(axis='y', alpha=0.3)
    
    plt.suptitle('四种模型预测精度对比', fontsize=FONT_SIZE_SUPTITLE,
                fontfamily=CHINESE_FONT, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'figure1_model_accuracy_comparison.png', dpi=DPI, bbox_inches='tight')
    plt.close()
    print("已生成: figure1_model_accuracy_comparison.png")

def figure2_time_scale_impact(df):
    """
    图2：时间尺度对预测性能的影响
    数据来源：experiment_summary.csv
    - 序列长度影响：按seq_len分组，计算各模型的平均R²值
    - 预测长度影响：按pred_len分组，计算各模型的平均R²值
    展示2个子图：序列长度影响、预测长度影响
    """
    fig, axes = plt.subplots(2, 1, figsize=(10, 10))
    
    # 子图1：序列长度影响
    ax1 = axes[0]
    for model in MODEL_NAMES:
        model_data = df[df['model'] == model]
        seq_r2 = model_data.groupby('seq_len')['R2'].mean()
        seq_r2 = seq_r2.reindex(SEQ_LENGTHS)
        
        ax1.plot(SEQ_LENGTHS, seq_r2.values, marker='o', linewidth=2,
                markersize=8, label=MODEL_NAMES_CN[model], color=COLORS[model])
    
    ax1.set_xlabel('序列长度 (步)', fontsize=FONT_SIZE_LABEL, fontfamily=CHINESE_FONT, fontweight='bold')
    ax1.set_ylabel('R²值', fontsize=FONT_SIZE_LABEL, fontfamily=CHINESE_FONT, fontweight='bold')
    ax1.set_title('(a) 序列长度对预测性能的影响', fontsize=FONT_SIZE_TITLE,
                 fontfamily=CHINESE_FONT, fontweight='bold', loc='left')
    ax1.set_xticks(SEQ_LENGTHS)
    ax1.set_xticklabels(SEQ_LABELS, fontsize=FONT_SIZE_TICK, fontfamily=CHINESE_FONT, fontweight='bold')
    ax1.legend(fontsize=FONT_SIZE_LEGEND, prop={'family': CHINESE_FONT, 'weight': 'bold'})
    set_ticklabels_bold(ax1)
    ax1.grid(alpha=0.3)
    
    # 子图2：预测长度影响
    ax2 = axes[1]
    for model in MODEL_NAMES:
        model_data = df[df['model'] == model]
        pred_r2 = model_data.groupby('pred_len')['R2'].mean()
        pred_r2 = pred_r2.reindex(PRED_LENGTHS)
        
        ax2.plot(PRED_LENGTHS, pred_r2.values, marker='s', linewidth=2,
                markersize=8, label=MODEL_NAMES_CN[model], color=COLORS[model])
    
    ax2.set_xlabel('预测长度 (步)', fontsize=FONT_SIZE_LABEL, fontfamily=CHINESE_FONT, fontweight='bold')
    ax2.set_ylabel('R²值', fontsize=FONT_SIZE_LABEL, fontfamily=CHINESE_FONT, fontweight='bold')
    ax2.set_title('(b) 预测长度对预测性能的影响', fontsize=FONT_SIZE_TITLE,
                 fontfamily=CHINESE_FONT, fontweight='bold', loc='left')
    ax2.set_xticks(PRED_LENGTHS)
    ax2.set_xticklabels(PRED_LABELS, fontsize=FONT_SIZE_TICK, fontfamily=CHINESE_FONT, fontweight='bold')
    ax2.legend(fontsize=FONT_SIZE_LEGEND, prop={'family': CHINESE_FONT, 'weight': 'bold'})
    set_ticklabels_bold(ax2)
    ax2.grid(alpha=0.3)
    
    plt.suptitle('时间尺度对预测性能的影响', fontsize=FONT_SIZE_SUPTITLE,
                fontfamily=CHINESE_FONT, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'figure2_time_scale_impact.png', dpi=DPI, bbox_inches='tight')
    plt.close()
    print("已生成: figure2_time_scale_impact.png")

def figure3_heatmap_comparison(df):
    """
    图3：时间尺度组合热力图对比
    数据来源：experiment_summary.csv
    - 对每个模型，创建seq_len × pred_len的R²值矩阵
    - 使用pivot_table创建透视表：df.pivot_table(values='R2', index='pred_len', columns='seq_len')
    展示4个子图：每个模型一个热力图
    """
    fig, axes = plt.subplots(4, 1, figsize=(10, 16))
    
    for idx, model in enumerate(MODEL_NAMES):
        ax = axes[idx]
        model_data = df[df['model'] == model]
        
        # 创建透视表
        pivot = model_data.pivot_table(values='R2', index='pred_len',
                                      columns='seq_len', aggfunc='mean')
        pivot = pivot.reindex(PRED_LENGTHS, columns=SEQ_LENGTHS)
        
        # 绘制热力图
        im = ax.imshow(pivot.values, cmap='RdYlGn', aspect='auto', vmin=-0.5, vmax=1.0)
        
        # 设置刻度
        ax.set_xticks(range(len(SEQ_LENGTHS)))
        ax.set_xticklabels(SEQ_LABELS, fontsize=FONT_SIZE_TICK-1, fontfamily=CHINESE_FONT, fontweight='bold')
        ax.set_yticks(range(len(PRED_LENGTHS)))
        ax.set_yticklabels(PRED_LABELS, fontsize=FONT_SIZE_TICK-1, fontfamily=CHINESE_FONT, fontweight='bold')
        
        # 添加数值
        for i in range(len(PRED_LENGTHS)):
            for j in range(len(SEQ_LENGTHS)):
                text = ax.text(j, i, f'{pivot.values[i, j]:.2f}',
                             ha="center", va="center", color="black",
                             fontsize=FONT_SIZE_TICK-2, fontfamily=ENGLISH_FONT, fontweight='bold')
        
        ax.set_xlabel('序列长度', fontsize=FONT_SIZE_LABEL, fontfamily=CHINESE_FONT, fontweight='bold')
        ax.set_ylabel('预测长度', fontsize=FONT_SIZE_LABEL, fontfamily=CHINESE_FONT, fontweight='bold')
        ax.set_title(f'({chr(97+idx)}) {MODEL_NAMES_CN[model]}', fontsize=FONT_SIZE_TITLE,
                    fontfamily=CHINESE_FONT, fontweight='bold', loc='left')
        set_ticklabels_bold(ax)
        
        # 为每个子图添加颜色条
        cbar = plt.colorbar(im, ax=ax, orientation='vertical', pad=0.02)
        cbar.set_label('R²值', fontsize=FONT_SIZE_LABEL-2, fontfamily=CHINESE_FONT, fontweight='bold')
    
    plt.suptitle('时间尺度组合热力图（R²值）', fontsize=FONT_SIZE_SUPTITLE,
                fontfamily=CHINESE_FONT, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'figure3_heatmap_comparison.png', dpi=DPI, bbox_inches='tight')
    plt.close()
    print("已生成: figure3_heatmap_comparison.png")

def figure4_tradeoff_analysis(df):
    """
    图4：精度-效率权衡分析
    数据来源：experiment_summary.csv
    - 散点图：inference_time_ms vs R2（所有64个实验点）
    - 性能对比：推理时间、内存占用、训练时间的平均值对比
    展示2个子图：散点图、性能对比柱状图
    """
    fig, axes = plt.subplots(2, 1, figsize=(10, 10))
    
    # 子图1：精度-效率散点图
    ax1 = axes[0]
    markers = ['o', 's', '^', 'D']
    for idx, model in enumerate(MODEL_NAMES):
        model_data = df[df['model'] == model]
        ax1.scatter(model_data['inference_time_ms'], model_data['R2'],
                   s=100, alpha=0.6, label=MODEL_NAMES_CN[model],
                   color=COLORS[model], marker=markers[idx], edgecolors='black', linewidths=0.5)
    
    ax1.set_xlabel('推理时间 (ms)', fontsize=FONT_SIZE_LABEL, fontfamily=CHINESE_FONT, fontweight='bold')
    ax1.set_ylabel('R²值', fontsize=FONT_SIZE_LABEL, fontfamily=CHINESE_FONT, fontweight='bold')
    ax1.set_title('(a) 精度-效率权衡散点图', fontsize=FONT_SIZE_TITLE,
                 fontfamily=CHINESE_FONT, fontweight='bold', loc='left')
    ax1.legend(fontsize=FONT_SIZE_LEGEND, prop={'family': CHINESE_FONT, 'weight': 'bold'})
    set_ticklabels_bold(ax1)
    ax1.grid(alpha=0.3)
    
    # 子图2：推理时间对比（突出效率差异）
    ax2 = axes[1]
    model_time = df.groupby('model')['inference_time_ms'].mean().sort_values(ascending=True)
    
    # 统一使用垂直柱状图
    bars = ax2.bar(range(len(model_time)), model_time.values,
                   color=[COLORS[m] for m in model_time.index])
    
    for i, (bar, val) in enumerate(zip(bars, model_time.values)):
        ax2.text(bar.get_x() + bar.get_width()/2, val + val*0.02,
               f'{val:.2f} ms', ha='center', va='bottom',
               fontsize=FONT_SIZE_TICK, fontfamily=ENGLISH_FONT, fontweight='bold')
    
    ax2.set_xticks(range(len(model_time)))
    ax2.set_xticklabels([MODEL_NAMES_CN[m] for m in model_time.index],
                        fontsize=FONT_SIZE_LABEL, fontfamily=CHINESE_FONT, fontweight='bold')
    ax2.set_ylabel('推理时间 (ms)', fontsize=FONT_SIZE_LABEL, fontfamily=CHINESE_FONT, fontweight='bold')
    ax2.set_title('(b) 模型推理时间对比', fontsize=FONT_SIZE_TITLE,
                 fontfamily=CHINESE_FONT, fontweight='bold', loc='left')
    set_ticklabels_bold(ax2)
    ax2.grid(axis='y', alpha=0.3)
    
    plt.suptitle('精度-效率权衡分析', fontsize=FONT_SIZE_SUPTITLE,
                fontfamily=CHINESE_FONT, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'figure4_tradeoff_analysis.png', dpi=DPI, bbox_inches='tight')
    plt.close()
    print("已生成: figure4_tradeoff_analysis.png")

def figure5_prediction_effects():
    """
    图5：关键参数预测效果对比
    数据来源：PKL文件（如Transformer_60_1_results.pkl等）
    - y_true_inv：真实值（反归一化后的数据）
    - y_pred_inv：预测值（反归一化后的数据）
    - 特征索引：贯入度(0)、推进压力(1)、刀盘转速(18)
    - 样本范围：测试集的前500个样本
    展示3个子图：贯入度、推进压力、刀盘转速的预测效果
    """
    seq_len = PKL_CONFIG['seq_len']
    pred_len = PKL_CONFIG['pred_len']
    start, end = PKL_CONFIG['sample_range']
    
    # 从第一个模型的PKL文件中获取特征映射
    first_data = None
    for model in MODEL_NAMES:
        data = load_pkl_data(model, seq_len, pred_len)
        if data is not None:
            first_data = data
            break
    
    if first_data is None:
        print("警告: 无法加载PKL数据，跳过图5")
        return
    
    # 从PKL文件中获取特征映射（优先使用PKL文件中的，否则使用默认值）
    feature_mapping = first_data.get('feature_mapping', {}).get('key_features', PKL_CONFIG['feature_indices'])
    
    features = ['贯入度', '推进压力', '刀盘转速']
    feature_indices = [feature_mapping.get(f, PKL_CONFIG['feature_indices'].get(f, 0)) for f in features]
    
    # 获取特征单位（从config或PKL文件）
    feature_units = []
    for feat_name in features:
        feat_idx = feature_mapping.get(feat_name, PKL_CONFIG['feature_indices'].get(feat_name, 0))
        # 尝试从PKL文件的feature_names中获取单位信息，或使用默认值
        if 'feature_names' in first_data:
            # 这里可以根据实际需要添加单位映射
            unit_map = {'贯入度': '(mm/min)', '推进压力': '(MPa)', '刀盘转速': '(r/min)'}
            feature_units.append(unit_map.get(feat_name, ''))
        else:
            unit_map = {'贯入度': '(mm/min)', '推进压力': '(MPa)', '刀盘转速': '(r/min)'}
            feature_units.append(unit_map.get(feat_name, ''))
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 12))
    
    for idx, (feature, feat_idx, unit) in enumerate(zip(features, feature_indices, feature_units)):
        ax = axes[idx]
        
        for model in MODEL_NAMES:
            data = load_pkl_data(model, seq_len, pred_len)
            if data is None:
                continue
            
            y_true = data['y_true_inv'][start:end, feat_idx]
            y_pred = data['y_pred_inv'][start:end, feat_idx]
            
            x = range(len(y_true))
            ax.plot(x, y_true, '--', linewidth=1.5, alpha=0.7,
                   label=f'{MODEL_NAMES_CN[model]} (真实值)', color=COLORS[model])
            ax.plot(x, y_pred, '-', linewidth=2,
                   label=f'{MODEL_NAMES_CN[model]} (预测值)', color=COLORS[model])
        
        ax.set_xlabel('样本索引', fontsize=FONT_SIZE_LABEL, fontfamily=CHINESE_FONT, fontweight='bold')
        ax.set_ylabel(f'{feature} {unit}', fontsize=FONT_SIZE_LABEL, fontfamily=CHINESE_FONT, fontweight='bold')
        ax.set_title(f'({chr(97+idx)}) {feature}', fontsize=FONT_SIZE_TITLE,
                    fontfamily=CHINESE_FONT, fontweight='bold', loc='left')
        ax.legend(fontsize=FONT_SIZE_LEGEND-2, prop={'family': CHINESE_FONT, 'weight': 'bold'}, ncol=2)
        set_ticklabels_bold(ax)
        ax.grid(alpha=0.3)
    
    plt.suptitle('关键参数预测效果对比', fontsize=FONT_SIZE_SUPTITLE,
                fontfamily=CHINESE_FONT, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'figure5_prediction_effects.png', dpi=DPI, bbox_inches='tight')
    plt.close()
    print("已生成: figure5_prediction_effects.png")

def figure6_computation_performance(df):
    """
    图6：模型计算性能对比
    数据来源：experiment_summary.csv
    - 推理时间：df.groupby('model')['inference_time_ms'].mean()
    - 内存占用：df.groupby('model')['memory_usage_mb'].mean()
    - 训练时间：df.groupby('model')['training_time_s'].mean()
    展示3个子图：推理时间、内存占用、训练时间
    """
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))
    
    metrics = ['inference_time_ms', 'memory_usage_mb', 'training_time_s']
    metric_names = ['推理时间', '内存占用', '训练时间']
    metric_units = ['(ms)', '(MB)', '(秒)']
    
    for idx, (metric, name, unit) in enumerate(zip(metrics, metric_names, metric_units)):
        ax = axes[idx]
        model_metric = df.groupby('model')[metric].mean().sort_values(ascending=(idx < 2))
        
        bars = ax.bar(range(len(model_metric)), model_metric.values,
                     color=[COLORS[m] for m in model_metric.index])
        
        # 添加数值标签
        for i, (bar, val) in enumerate(zip(bars, model_metric.values)):
            if idx == 2:  # 训练时间用秒
                label_text = f'{val:.1f} s'
            elif idx == 1:  # 内存用MB
                label_text = f'{val:.0f} MB'
            else:  # 推理时间用ms
                label_text = f'{val:.2f} ms'
            
            ax.text(bar.get_x() + bar.get_width()/2, val + val*0.02,
                   label_text, ha='center', va='bottom',
                   fontsize=FONT_SIZE_TICK, fontfamily=ENGLISH_FONT, fontweight='bold')
        
        ax.set_xticks(range(len(model_metric)))
        ax.set_xticklabels([MODEL_NAMES_CN[m] for m in model_metric.index],
                          fontsize=FONT_SIZE_LABEL, fontfamily=CHINESE_FONT, fontweight='bold')
        ax.set_ylabel(f'{name} {unit}', fontsize=FONT_SIZE_LABEL, fontfamily=CHINESE_FONT, fontweight='bold')
        ax.set_title(f'({chr(97+idx)}) {name}', fontsize=FONT_SIZE_TITLE,
                    fontfamily=CHINESE_FONT, fontweight='bold', loc='left')
        set_ticklabels_bold(ax)
        ax.grid(axis='y', alpha=0.3)
    
    plt.suptitle('模型计算性能对比', fontsize=FONT_SIZE_SUPTITLE,
                fontfamily=CHINESE_FONT, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'figure6_computation_performance.png', dpi=DPI, bbox_inches='tight')
    plt.close()
    print("已生成: figure6_computation_performance.png")

def figure7_error_distribution():
    """
    图7：多参数预测误差分布箱线图
    数据来源：PKL文件（如Transformer_60_1_results.pkl等）
    - 对每个模型，计算30个参数的MSE值
    - MSE = mean((y_true_inv[:, feat_idx] - y_pred_inv[:, feat_idx])^2)
    - 展示30个参数的误差分布
    """
    seq_len = PKL_CONFIG['seq_len']
    pred_len = PKL_CONFIG['pred_len']
    
    all_errors = {model: [] for model in MODEL_NAMES}
    
    # 加载第一个模型获取特征数量
    first_data = None
    for model in MODEL_NAMES:
        data = load_pkl_data(model, seq_len, pred_len)
        if data is not None:
            first_data = data
            break
    
    if first_data is None:
        print("警告: 无法加载PKL数据，跳过图7")
        return
    
    n_features = first_data['y_true_inv'].shape[1]
    
    # 计算每个参数的MSE
    for model in MODEL_NAMES:
        data = load_pkl_data(model, seq_len, pred_len)
        if data is None:
            continue
        
        y_true = data['y_true_inv']
        y_pred = data['y_pred_inv']
        
        # 计算每个特征的MSE
        for feat_idx in range(min(30, n_features)):
            mse = np.mean((y_true[:, feat_idx] - y_pred[:, feat_idx]) ** 2)
            all_errors[model].append(mse)
    
    # 绘制箱线图
    fig, ax = plt.subplots(figsize=(16, 6))
    
    # 准备数据：每个参数一个箱线图，包含4个模型的数据
    box_data = []
    for feat_idx in range(min(30, n_features)):
        feat_errors = []
        for model in MODEL_NAMES:
            if len(all_errors[model]) > feat_idx:
                feat_errors.append(all_errors[model][feat_idx])
        box_data.append(feat_errors)
    
    bp = ax.boxplot(box_data, labels=[f'参数{i+1}' for i in range(len(box_data))],
                   patch_artist=True, widths=0.6, showmeans=True)
    
    # 设置颜色
    for patch in bp['boxes']:
        patch.set_facecolor('#3498db')
        patch.set_alpha(0.7)
    
    ax.set_xlabel('参数编号', fontsize=FONT_SIZE_LABEL, fontfamily=CHINESE_FONT, fontweight='bold')
    ax.set_ylabel('MSE', fontsize=FONT_SIZE_LABEL, fontfamily=ENGLISH_FONT, fontweight='bold')
    ax.set_title('多参数预测误差分布箱线图', fontsize=FONT_SIZE_TITLE,
                fontfamily=CHINESE_FONT, fontweight='bold')
    ax.tick_params(axis='x', rotation=45, labelsize=FONT_SIZE_TICK-2)
    set_ticklabels_bold(ax)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'figure7_error_distribution.png', dpi=DPI, bbox_inches='tight')
    plt.close()
    print("已生成: figure7_error_distribution.png")

def figure8_radar_chart(df):
    """
    图8：模型综合性能雷达图
    数据来源：experiment_summary.csv
    - 选择4个维度：R²、推理速度、内存占用、训练时间
    - 对每个模型计算平均值，然后归一化到[0,1]范围
    - R²越大越好，其他越小越好（需要反向归一化）
    """
    from math import pi
    
    metrics = ['R2', 'inference_time_ms', 'memory_usage_mb', 'training_time_s']
    metric_names = ['R²', '推理速度', '内存占用', '训练时间']
    
    # 归一化函数
    def normalize_r2(x, min_val, max_val):
        return (x - min_val) / (max_val - min_val) if max_val != min_val else 0.5
    
    def normalize_others(x, min_val, max_val):
        return 1 - (x - min_val) / (max_val - min_val) if max_val != min_val else 0.5
    
    fig, axes = plt.subplots(4, 1, figsize=(10, 16), subplot_kw=dict(projection='polar'))
    
    # 计算归一化范围
    ranges = {}
    for metric in metrics:
        ranges[metric] = (df[metric].min(), df[metric].max())
    
    for idx, model in enumerate(MODEL_NAMES):
        ax = axes[idx]
        model_data = df[df['model'] == model]
        
        # 计算平均值并归一化
        values = []
        for metric in metrics:
            mean_val = model_data[metric].mean()
            if metric == 'R2':
                norm_val = normalize_r2(mean_val, ranges[metric][0], ranges[metric][1])
            else:
                norm_val = normalize_others(mean_val, ranges[metric][0], ranges[metric][1])
            values.append(norm_val)
        
        # 闭合图形
        values += values[:1]
        
        # 角度
        angles = [n / float(len(metrics)) * 2 * pi for n in range(len(metrics))]
        angles += angles[:1]
        
        ax.plot(angles, values, 'o-', linewidth=2, color=COLORS[model], label=MODEL_NAMES_CN[model])
        ax.fill(angles, values, alpha=0.25, color=COLORS[model])
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metric_names, fontsize=FONT_SIZE_TICK, fontfamily=CHINESE_FONT, fontweight='bold')
        ax.set_ylim(0, 1)
        ax.set_title(f'({chr(97+idx)}) {MODEL_NAMES_CN[model]}', fontsize=FONT_SIZE_TITLE,
                     fontfamily=CHINESE_FONT, fontweight='bold', pad=20)
        set_ticklabels_bold(ax)
        ax.grid(True)
    
    plt.suptitle('模型综合性能雷达图', fontsize=FONT_SIZE_SUPTITLE,
                fontfamily=CHINESE_FONT, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'figure8_radar_chart.png', dpi=DPI, bbox_inches='tight')
    plt.close()
    print("已生成: figure8_radar_chart.png")

# =============================================================================
# 主函数
# =============================================================================

def main():
    """主函数"""
    print("="*60)
    print("论文图片生成脚本 - 生成8个最重要的图片")
    print("="*60)
    
    # 设置字体
    setup_fonts()
    
    # 加载数据
    df = load_data()
    
    # 生成8个最重要的图片
    print("\n开始生成图片...")
    figure1_model_accuracy_comparison(df)
    figure2_time_scale_impact(df)
    figure3_heatmap_comparison(df)
    figure4_tradeoff_analysis(df)
    figure5_prediction_effects()
    figure6_computation_performance(df)
    figure7_error_distribution()
    figure8_radar_chart(df)
    
    print("\n" + "="*60)
    print(f"所有图片已生成到: {OUTPUT_DIR}")
    print("共生成8张图片：")
    print("  图1：四种模型预测精度对比（4子图）")
    print("  图2：时间尺度对预测性能的影响（2子图）")
    print("  图3：时间尺度组合热力图（4子图）")
    print("  图4：精度-效率权衡分析（2子图）")
    print("  图5：关键参数预测效果对比（3子图）")
    print("  图6：模型计算性能对比（3子图）")
    print("  图7：多参数预测误差分布箱线图")
    print("  图8：模型综合性能雷达图（4子图）")
    print("="*60)

if __name__ == '__main__':
    main()

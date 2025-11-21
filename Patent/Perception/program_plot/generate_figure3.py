"""
专利附图3：数据预处理流程示意图生成脚本
包括：异常值检测（箱线图）、信号去噪（小波变换）、特征降维（PCA）
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import font_manager
import pywt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 设置中文字体为宋体
plt.rcParams['font.sans-serif'] = ['SimSun', 'STSong', 'Songti SC', '宋体']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 11  # 统一字体大小

# 设置图形参数（符合专利附图要求：黑白线条图）
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.format'] = 'png'
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['axes.linewidth'] = 1.0  # 简化线条
plt.rcParams['grid.alpha'] = 0.3  # 简化网格

# 生成示例数据
np.random.seed(42)

def generate_sample_data():
    """生成示例盾构监测数据"""
    n_samples = 200
    time = np.linspace(0, 20, n_samples)
    
    # 模拟刀盘扭矩数据（含异常值和噪声）
    torque_base = 1000 + 200 * np.sin(2 * np.pi * time / 5)
    torque_noise = np.random.normal(0, 50, n_samples)
    torque = torque_base + torque_noise
    
    # 添加异常值
    anomaly_indices = [50, 120, 180]
    torque[anomaly_indices] = [2500, 300, 2800]
    
    return time, torque

def plot_boxplot_anomaly_detection(time, data, save_path='figure3a_boxplot.png'):
    """
    绘制箱线图异常值检测示意图（简化版）
    图3a：异常值检测（箱线图法）
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    # 计算IQR和异常值阈值
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # 绘制箱线图
    bp = ax.boxplot([data], vert=True, patch_artist=True, widths=0.5, 
                     showmeans=False, meanline=False)
    
    # 设置箱线图样式（黑白）
    bp['boxes'][0].set_facecolor('white')
    bp['boxes'][0].set_edgecolor('black')
    bp['boxes'][0].set_linewidth(1.5)
    for whisker in bp['whiskers']:
        whisker.set_color('black')
        whisker.set_linewidth(1.5)
    for cap in bp['caps']:
        cap.set_color('black')
        cap.set_linewidth(1.5)
    bp['medians'][0].set_color('black')
    bp['medians'][0].set_linewidth(1.5)
    bp['fliers'][0].set_marker('x')
    bp['fliers'][0].set_markeredgecolor('black')
    bp['fliers'][0].set_markersize=6
    bp['fliers'][0].set_markeredgewidth=1.5
    
    # 标注关键值（简化）
    median_val = np.median(data)
    ax.text(1.15, Q1, f'Q1', ha='left', va='center', fontsize=10)
    ax.text(1.15, Q3, f'Q3', ha='left', va='center', fontsize=10)
    ax.text(1.15, median_val, f'中位数', ha='left', va='center', fontsize=10)
    
    # 标注异常值上下界
    ax.axhline(y=upper_bound, color='black', linestyle='--', linewidth=1.0)
    ax.axhline(y=lower_bound, color='black', linestyle='--', linewidth=1.0)
    ax.text(1.25, upper_bound, f'上界\nQ3+1.5×IQR', ha='left', va='center', fontsize=9)
    ax.text(1.25, lower_bound, f'下界\nQ1-1.5×IQR', ha='left', va='center', fontsize=9)
    
    ax.set_ylabel('监测参数值', fontsize=11)
    ax.set_title('(a) 异常值检测：箱线图法', fontsize=12, fontweight='bold')
    ax.set_xticks([1])
    ax.set_xticklabels(['监测数据'])
    ax.grid(True, alpha=0.2, linestyle='--', axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, format='png', bbox_inches='tight', facecolor='white')
    print(f"箱线图已保存至: {save_path}")
    plt.close()

def plot_wavelet_denoising(time, data, save_path='figure3b_wavelet.png'):
    """
    绘制小波变换去噪示意图（简化版）
    图3b：信号去噪（小波变换）
    """
    fig, axes = plt.subplots(2, 1, figsize=(8, 6))
    
    # 小波分解
    wavelet = 'db10'  # Daubechies 10小波
    coeffs = pywt.wavedec(data, wavelet, level=1)
    cA, cD = coeffs  # 近似系数（低频）和细节系数（高频）
    
    # 重构近似系数（去噪）
    data_denoised = pywt.waverec([cA, np.zeros_like(cD)], wavelet)
    
    # 确保长度一致
    if len(data_denoised) > len(time):
        data_denoised = data_denoised[:len(time)]
    elif len(data_denoised) < len(time):
        time_denoised = time[:len(data_denoised)]
    else:
        time_denoised = time
    
    # 上图：原始信号和去噪后信号对比
    axes[0].plot(time, data, 'k--', linewidth=1.0, alpha=0.6, label='原始信号（含噪声）')
    axes[0].plot(time_denoised, data_denoised, 'k-', linewidth=1.5, label='去噪后信号')
    axes[0].set_xlabel('时间', fontsize=11)
    axes[0].set_ylabel('幅值', fontsize=11)
    axes[0].set_title('(b) 信号去噪：离散小波变换（db10小波基）', fontsize=12, fontweight='bold')
    axes[0].legend(loc='upper right', fontsize=10)
    axes[0].grid(True, alpha=0.2, linestyle='--')
    
    # 下图：小波分解示意（简化）
    axes[1].plot(time[:len(cA)], cA, 'k-', linewidth=1.5, label='近似系数（低频，保留）')
    axes[1].plot(time[:len(cD)], cD, 'k--', linewidth=1.0, alpha=0.5, label='细节系数（高频，抑制）')
    axes[1].axhline(y=0, color='gray', linestyle=':', linewidth=0.5)
    axes[1].set_xlabel('时间', fontsize=11)
    axes[1].set_ylabel('小波系数', fontsize=11)
    axes[1].set_title('小波分解：提取低频主趋势，抑制高频噪声', fontsize=11)
    axes[1].legend(loc='upper right', fontsize=9)
    axes[1].grid(True, alpha=0.2, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(save_path, format='png', bbox_inches='tight', facecolor='white')
    print(f"小波变换图已保存至: {save_path}")
    plt.close()

def plot_pca_dimensionality_reduction(save_path='figure3c_pca.png'):
    """
    绘制PCA特征降维示意图（简化版）
    图3c：特征降维（主成分分析）
    """
    # 生成高维示例数据（模拟多个监测参数）
    np.random.seed(42)
    n_samples = 80
    n_features = 8  # 8个监测参数
    
    # 生成相关的高维数据
    data_high = np.random.randn(n_samples, n_features)
    data_high[:, 1] = 0.7 * data_high[:, 0] + 0.3 * np.random.randn(n_samples)
    data_high[:, 2] = 0.5 * data_high[:, 0] + 0.5 * np.random.randn(n_samples)
    
    # 标准化
    scaler = StandardScaler()
    data_high_scaled = scaler.fit_transform(data_high)
    
    # PCA降维
    pca = PCA(n_components=2)  # 降至2维用于可视化
    data_low = pca.fit_transform(data_high_scaled)
    
    # 计算累计贡献率
    pca_full = PCA()
    pca_full.fit(data_high_scaled)
    cumsum_ratio = np.cumsum(pca_full.explained_variance_ratio_)
    n_components_95 = np.where(cumsum_ratio >= 0.95)[0][0] + 1
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    # 左图：高维特征空间（3D投影，简化）
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.scatter(data_high_scaled[:, 0], data_high_scaled[:, 1], 
                data_high_scaled[:, 2], c='black', s=20, alpha=0.6)
    ax1.set_xlabel('特征1', fontsize=10)
    ax1.set_ylabel('特征2', fontsize=10)
    ax1.set_zlabel('特征3', fontsize=10)
    ax1.set_title(f'高维特征空间\n({n_features}维)', fontsize=11, fontweight='bold')
    
    # 中图：主成分贡献率（简化）
    ax2 = axes[1]
    components = np.arange(1, min(6, len(pca_full.explained_variance_ratio_)) + 1)
    ax2.bar(components, pca_full.explained_variance_ratio_[:len(components)], 
           color='black', alpha=0.5, edgecolor='black', linewidth=1.0)
    ax2.plot(components, cumsum_ratio[:len(components)], 'ko-', 
            linewidth=1.5, markersize=4, label='累计贡献率')
    ax2.axhline(y=0.95, color='black', linestyle='--', linewidth=1.0)
    ax2.set_xlabel('主成分序号', fontsize=10)
    ax2.set_ylabel('贡献率', fontsize=10)
    ax2.set_title('主成分贡献率', fontsize=11, fontweight='bold')
    ax2.legend(loc='lower right', fontsize=9)
    ax2.grid(True, alpha=0.2, linestyle='--', axis='y')
    ax2.text(3, 0.5, f'前{n_components_95}个主成分\n累计贡献率≥95%', 
            ha='center', va='center', fontsize=9,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='black'))
    
    # 右图：降维后的低维特征空间（简化）
    ax3 = axes[2]
    ax3.scatter(data_low[:, 0], data_low[:, 1], c='black', s=30, alpha=0.6)
    ax3.set_xlabel(f'第一主成分 (PC1)', fontsize=10)
    ax3.set_ylabel(f'第二主成分 (PC2)', fontsize=10)
    ax3.set_title('降维后特征空间\n(2维主成分)', fontsize=11, fontweight='bold')
    ax3.grid(True, alpha=0.2, linestyle='--')
    
    # 添加箭头表示降维过程
    fig.text(0.33, 0.5, '→', fontsize=24, ha='center', va='center', fontweight='bold')
    fig.text(0.67, 0.5, '→', fontsize=24, ha='center', va='center', fontweight='bold')
    
    plt.suptitle('(c) 特征降维：主成分分析（PCA）', fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, format='png', bbox_inches='tight', facecolor='white')
    print(f"PCA降维图已保存至: {save_path}")
    plt.close()

def plot_combined_figure3(time, data, save_path='figure3_combined.png'):
    """
    绘制完整的图3：数据预处理流程示意图（三个子图组合，简化版）
    """
    fig = plt.figure(figsize=(14, 8))
    
    # 图3a：异常值检测（箱线图）
    ax1 = plt.subplot(2, 3, 1)
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    bp = ax1.boxplot([data], vert=True, patch_artist=True, widths=0.5)
    bp['boxes'][0].set_facecolor('white')
    bp['boxes'][0].set_edgecolor('black')
    bp['boxes'][0].set_linewidth(1.5)
    for whisker in bp['whiskers']:
        whisker.set_color('black')
        whisker.set_linewidth(1.5)
    for cap in bp['caps']:
        cap.set_color('black')
        cap.set_linewidth(1.5)
    bp['medians'][0].set_color('black')
    bp['medians'][0].set_linewidth(1.5)
    bp['fliers'][0].set_marker('x')
    bp['fliers'][0].set_markeredgecolor('black')
    
    ax1.axhline(y=upper_bound, color='black', linestyle='--', linewidth=1.0)
    ax1.axhline(y=lower_bound, color='black', linestyle='--', linewidth=1.0)
    ax1.set_ylabel('监测参数值', fontsize=10)
    ax1.set_title('(a) 异常值检测\n箱线图法', fontsize=11, fontweight='bold')
    ax1.set_xticks([1])
    ax1.set_xticklabels(['数据'])
    ax1.grid(True, alpha=0.2, linestyle='--', axis='y')
    
    # 图3b：小波去噪
    ax2 = plt.subplot(2, 3, 2)
    wavelet = 'db10'
    coeffs = pywt.wavedec(data, wavelet, level=1)
    cA, cD = coeffs
    data_denoised = pywt.waverec([cA, np.zeros_like(cD)], wavelet)
    if len(data_denoised) > len(time):
        data_denoised = data_denoised[:len(time)]
    time_denoised = time[:len(data_denoised)]
    
    ax2.plot(time, data, 'k--', linewidth=1.0, alpha=0.5, label='原始信号')
    ax2.plot(time_denoised, data_denoised, 'k-', linewidth=1.5, label='去噪后')
    ax2.set_xlabel('时间', fontsize=10)
    ax2.set_ylabel('幅值', fontsize=10)
    ax2.set_title('(b) 信号去噪\n小波变换', fontsize=11, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.2, linestyle='--')
    
    # 图3c：PCA降维（简化）
    ax3 = plt.subplot(2, 3, 3)
    np.random.seed(42)
    data_high = np.random.randn(60, 8)
    data_high[:, 1] = 0.7 * data_high[:, 0] + 0.3 * np.random.randn(60)
    scaler = StandardScaler()
    data_high_scaled = scaler.fit_transform(data_high)
    pca = PCA(n_components=2)
    data_low = pca.fit_transform(data_high_scaled)
    
    ax3.scatter(data_low[:, 0], data_low[:, 1], c='black', s=25, alpha=0.6)
    ax3.set_xlabel('第一主成分', fontsize=10)
    ax3.set_ylabel('第二主成分', fontsize=10)
    ax3.set_title('(c) 特征降维\n主成分分析', fontsize=11, fontweight='bold')
    ax3.grid(True, alpha=0.2, linestyle='--')
    
    # 添加流程箭头（一行显示，增加与图的距离）
    plt.tight_layout(rect=[0, 0.15, 1, 0.98])  # 底部留出更多空间
    fig.text(0.5, 0.08, '原始数据 → 异常检测 → 去噪 → 降维 → 预处理后数据', 
            ha='center', fontsize=12, fontweight='bold')
    
    plt.savefig(save_path, format='png', bbox_inches='tight', facecolor='white')
    print(f"完整图3已保存至: {save_path}")
    plt.close()

def main():
    """主函数"""
    print("=" * 60)
    print("专利附图3：数据预处理流程示意图生成")
    print("=" * 60)
    
    # 生成示例数据
    print("\n1. 生成示例盾构监测数据...")
    time, torque_data = generate_sample_data()
    
    # 绘制各个子图
    print("\n2. 绘制图3a：异常值检测（箱线图法）...")
    plot_boxplot_anomaly_detection(time, torque_data, 'figure3a_boxplot.png')
    
    print("\n3. 绘制图3b：信号去噪（小波变换）...")
    plot_wavelet_denoising(time, torque_data, 'figure3b_wavelet.png')
    
    print("\n4. 绘制图3c：特征降维（PCA）...")
    plot_pca_dimensionality_reduction('figure3c_pca.png')
    
    print("\n5. 绘制完整图3：数据预处理流程示意图...")
    plot_combined_figure3(time, torque_data, 'figure3_combined.png')
    
    print("\n" + "=" * 60)
    print("所有图表生成完成！")
    print("=" * 60)
    print("\n生成的文件：")
    print("  - figure3a_boxplot.png: 异常值检测箱线图")
    print("  - figure3b_wavelet.png: 小波变换去噪图")
    print("  - figure3c_pca.png: PCA特征降维图")
    print("  - figure3_combined.png: 完整组合图")
    print("\n提示：这些图可以直接用于专利附图，或根据需要进行进一步编辑。")

if __name__ == '__main__':
    main()


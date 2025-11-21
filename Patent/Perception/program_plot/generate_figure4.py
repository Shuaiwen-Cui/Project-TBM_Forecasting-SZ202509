"""
专利附图4：特征解耦机制原理示意图生成脚本
展示主动量与被动量的区分与提取过程
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

# 设置中文字体为宋体
plt.rcParams['font.sans-serif'] = ['SimSun', 'STSong', 'Songti SC', '宋体']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 11

# 设置图形参数（符合专利附图要求：黑白线条图）
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.format'] = 'png'
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['axes.linewidth'] = 1.0

def plot_feature_decoupling(save_path='figure4_decoupling.png'):
    """
    绘制特征解耦机制原理示意图
    图4：特征解耦机制原理示意图
    """
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    ax.axis('off')
    
    # 定义位置参数
    x_start = 0.1
    x_mid = 0.5
    x_end = 0.9
    y_top = 0.85
    y_mid = 0.5
    y_bottom = 0.15
    
    # 1. 左侧：解耦前（混合特征）
    box_before = FancyBboxPatch((x_start-0.08, y_mid-0.15), 0.16, 0.3,
                                boxstyle="round,pad=0.02", 
                                edgecolor='black', facecolor='white', linewidth=1.5)
    ax.add_patch(box_before)
    ax.text(x_start, y_mid+0.1, '监测参数', ha='center', va='center', 
            fontsize=12, fontweight='bold')
    
    # 绘制混合特征示意（多个参数）
    param_names = ['推进速度', '刀盘转速', '刀盘扭矩', '推进压力', '贯入度']
    for i, name in enumerate(param_names):
        y_pos = y_mid - 0.05 + (len(param_names)-1-i) * 0.025
        ax.text(x_start-0.05, y_pos, '•', ha='left', va='center', fontsize=10)
        ax.text(x_start+0.02, y_pos, name, ha='left', va='center', fontsize=9)
    
    ax.text(x_start, y_mid-0.25, '（混合特征）', ha='center', va='center', 
            fontsize=10, style='italic')
    
    # 2. 中间：解耦过程
    # 箭头1：从监测参数到协方差计算
    arrow1 = FancyArrowPatch((x_start+0.08, y_mid), (x_mid-0.12, y_top),
                            arrowstyle='->', mutation_scale=20, 
                            linewidth=1.5, color='black')
    ax.add_patch(arrow1)
    
    # 协方差矩阵计算框
    box_cov = FancyBboxPatch((x_mid-0.12, y_top-0.08), 0.24, 0.16,
                             boxstyle="round,pad=0.02", 
                             edgecolor='black', facecolor='white', linewidth=1.5)
    ax.add_patch(box_cov)
    ax.text(x_mid, y_top, '协方差矩阵计算', ha='center', va='center', 
            fontsize=11, fontweight='bold')
    
    # 箭头2：从协方差到响应系数
    arrow2 = FancyArrowPatch((x_mid, y_top-0.08), (x_mid, y_mid+0.15),
                            arrowstyle='->', mutation_scale=20, 
                            linewidth=1.5, color='black')
    ax.add_patch(arrow2)
    
    # 响应系数分析框
    box_response = FancyBboxPatch((x_mid-0.12, y_mid-0.08), 0.24, 0.16,
                                  boxstyle="round,pad=0.02", 
                                  edgecolor='black', facecolor='white', linewidth=1.5)
    ax.add_patch(box_response)
    ax.text(x_mid, y_mid, '响应系数分析', ha='center', va='center', 
            fontsize=11, fontweight='bold')
    
    # 阈值判定框（菱形）
    threshold_x = x_mid
    threshold_y = y_mid - 0.15
    diamond = mpatches.RegularPolygon((threshold_x, threshold_y), 4, 
                                     radius=0.08, orientation=np.pi/4,
                                     edgecolor='black', facecolor='white', linewidth=1.5)
    ax.add_patch(diamond)
    ax.text(threshold_x, threshold_y, '阈值判定\n(0.7)', ha='center', va='center', 
            fontsize=10, fontweight='bold')
    
    # 箭头3：从阈值判定到主动量
    arrow3 = FancyArrowPatch((threshold_x-0.08, threshold_y-0.08), 
                            (x_end-0.12, y_top-0.05),
                            arrowstyle='->', mutation_scale=20, 
                            linewidth=1.5, color='black')
    ax.add_patch(arrow3)
    
    # 箭头4：从阈值判定到被动量
    arrow4 = FancyArrowPatch((threshold_x-0.08, threshold_y-0.08), 
                            (x_end-0.12, y_bottom+0.05),
                            arrowstyle='->', mutation_scale=20, 
                            linewidth=1.5, color='black')
    ax.add_patch(arrow4)
    
    # 3. 右侧：解耦后
    # 主动量分支
    box_active = FancyBboxPatch((x_end-0.12, y_top-0.12), 0.24, 0.24,
                                boxstyle="round,pad=0.02", 
                                edgecolor='black', facecolor='white', linewidth=1.5)
    ax.add_patch(box_active)
    ax.text(x_end, y_top+0.05, '主动量', ha='center', va='center', 
            fontsize=12, fontweight='bold')
    
    active_params = ['推进速度', '刀盘转速']
    for i, name in enumerate(active_params):
        y_pos = y_top - 0.02 - i * 0.04
        ax.text(x_end-0.08, y_pos, '•', ha='left', va='center', fontsize=10)
        ax.text(x_end-0.05, y_pos, name, ha='left', va='center', fontsize=9)
    
    ax.text(x_end, y_top-0.1, '受控制指令主导', ha='center', va='center', 
            fontsize=9, style='italic')
    
    # 被动量分支
    box_passive = FancyBboxPatch((x_end-0.12, y_bottom-0.12), 0.24, 0.24,
                                 boxstyle="round,pad=0.02", 
                                 edgecolor='black', facecolor='white', linewidth=1.5)
    ax.add_patch(box_passive)
    ax.text(x_end, y_bottom+0.05, '被动量', ha='center', va='center', 
            fontsize=12, fontweight='bold')
    
    passive_params = ['刀盘扭矩', '推进压力', '贯入度']
    for i, name in enumerate(passive_params):
        y_pos = y_bottom - 0.02 - i * 0.04
        ax.text(x_end-0.08, y_pos, '•', ha='left', va='center', fontsize=10)
        ax.text(x_end-0.05, y_pos, name, ha='left', va='center', fontsize=9)
    
    ax.text(x_end, y_bottom-0.1, '反映地层反馈', ha='center', va='center', 
            fontsize=9, style='italic')
    
    # 添加说明文字
    ax.text(x_mid, 0.02, '特征解耦机制：通过协方差分析与响应系数判定，实现地层特征与操控特征的有效分离', 
            ha='center', va='bottom', fontsize=10, 
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='black'))
    
    plt.tight_layout()
    plt.savefig(save_path, format='png', bbox_inches='tight', facecolor='white')
    print(f"特征解耦机制图已保存至: {save_path}")
    plt.close()

def plot_feature_decoupling_simple(save_path='figure4_decoupling_simple.png'):
    """
    绘制特征解耦机制原理示意图（简化版，更简洁）
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    ax.axis('off')
    
    # 定义位置参数
    x_start = 0.1
    x_cov = 0.35
    x_response = 0.5
    x_threshold = 0.65
    x_end = 0.9
    y_center = 0.5
    
    # 1. 监测参数（左侧）
    box_params = Rectangle((x_start-0.06, y_center-0.15), 0.12, 0.3,
                          edgecolor='black', facecolor='white', linewidth=1.5)
    ax.add_patch(box_params)
    ax.text(x_start, y_center+0.1, '监测参数', ha='center', va='center', 
            fontsize=11, fontweight='bold')
    ax.text(x_start, y_center, '推进速度\n刀盘转速\n刀盘扭矩\n推进压力\n贯入度', 
            ha='center', va='center', fontsize=9)
    ax.text(x_start, y_center-0.2, '（混合特征）', ha='center', va='center', 
            fontsize=9, style='italic')
    
    # 箭头1
    ax.arrow(x_start+0.06, y_center, x_cov-x_start-0.06, 0, 
            head_width=0.02, head_length=0.02, fc='black', ec='black', linewidth=1.5)
    
    # 2. 协方差矩阵计算
    box_cov = Rectangle((x_cov-0.06, y_center-0.1), 0.12, 0.2,
                       edgecolor='black', facecolor='white', linewidth=1.5)
    ax.add_patch(box_cov)
    ax.text(x_cov, y_center, '协方差\n矩阵计算', ha='center', va='center', 
            fontsize=10, fontweight='bold')
    
    # 箭头2
    ax.arrow(x_cov+0.06, y_center, x_response-x_cov-0.06, 0, 
            head_width=0.02, head_length=0.02, fc='black', ec='black', linewidth=1.5)
    
    # 3. 响应系数分析
    box_response = Rectangle((x_response-0.06, y_center-0.1), 0.12, 0.2,
                             edgecolor='black', facecolor='white', linewidth=1.5)
    ax.add_patch(box_response)
    ax.text(x_response, y_center, '响应系数\n分析', ha='center', va='center', 
            fontsize=10, fontweight='bold')
    
    # 箭头3
    ax.arrow(x_response+0.06, y_center, x_threshold-x_response-0.06, 0, 
            head_width=0.02, head_length=0.02, fc='black', ec='black', linewidth=1.5)
    
    # 4. 阈值判定（菱形）
    diamond = mpatches.RegularPolygon((x_threshold, y_center), 4, 
                                     radius=0.06, orientation=np.pi/4,
                                     edgecolor='black', facecolor='white', linewidth=1.5)
    ax.add_patch(diamond)
    ax.text(x_threshold, y_center, '阈值判定\n(0.7)', ha='center', va='center', 
            fontsize=9, fontweight='bold')
    
    # 箭头4和5：分支到主动量和被动量
    # 主动量（上方）
    ax.arrow(x_threshold+0.06, y_center+0.06, x_end-x_threshold-0.12, 0.15, 
            head_width=0.02, head_length=0.02, fc='black', ec='black', linewidth=1.5)
    
    box_active = Rectangle((x_end-0.08, y_center+0.2-0.08), 0.16, 0.16,
                          edgecolor='black', facecolor='white', linewidth=1.5)
    ax.add_patch(box_active)
    ax.text(x_end, y_center+0.2+0.02, '主动量', ha='center', va='center', 
            fontsize=11, fontweight='bold')
    ax.text(x_end, y_center+0.2-0.02, '推进速度\n刀盘转速\n受控制指令主导', 
            ha='center', va='center', fontsize=9)
    
    # 被动量（下方）
    ax.arrow(x_threshold+0.06, y_center-0.06, x_end-x_threshold-0.12, -0.15, 
            head_width=0.02, head_length=0.02, fc='black', ec='black', linewidth=1.5)
    
    box_passive = Rectangle((x_end-0.08, y_center-0.2-0.08), 0.16, 0.16,
                           edgecolor='black', facecolor='white', linewidth=1.5)
    ax.add_patch(box_passive)
    ax.text(x_end, y_center-0.2+0.02, '被动量', ha='center', va='center', 
            fontsize=11, fontweight='bold')
    ax.text(x_end, y_center-0.2-0.02, '刀盘扭矩\n推进压力\n贯入度\n反映地层反馈', 
            ha='center', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, format='png', bbox_inches='tight', facecolor='white')
    print(f"特征解耦机制图（简化版）已保存至: {save_path}")
    plt.close()

def main():
    """主函数"""
    print("=" * 60)
    print("专利附图4：特征解耦机制原理示意图生成")
    print("=" * 60)
    
    print("\n1. 绘制图4：特征解耦机制原理示意图...")
    plot_feature_decoupling('figure4_decoupling.png')
    
    print("\n2. 绘制图4（简化版）：特征解耦机制原理示意图...")
    plot_feature_decoupling_simple('figure4_decoupling_simple.png')
    
    print("\n" + "=" * 60)
    print("所有图表生成完成！")
    print("=" * 60)
    print("\n生成的文件：")
    print("  - figure4_decoupling.png: 特征解耦机制图（完整版）")
    print("  - figure4_decoupling_simple.png: 特征解耦机制图（简化版，推荐）")
    print("\n提示：这些图可以直接用于专利附图，或根据需要进行进一步编辑。")

if __name__ == '__main__':
    main()


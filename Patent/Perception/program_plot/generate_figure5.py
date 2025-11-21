"""
专利附图5：一维卷积神经网络（1D-CNN）结构示意图生成脚本
展示地层特征在网络中的逐层提取与输出分类机制
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle, FancyBboxPatch, FancyArrowPatch, Circle

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

def plot_1dcnn_structure(save_path='figure5_1dcnn.png'):
    """
    绘制一维卷积神经网络（1D-CNN）结构示意图
    图5：1D-CNN结构示意图
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 14))
    ax.axis('off')
    
    # 定义位置参数
    x_center = 0.5
    y_start = 0.95
    layer_height = 0.08
    arrow_length = 0.06
    box_width = 0.35
    box_height = 0.07
    
    current_y = y_start
    
    # 1. 输入层
    box_input = FancyBboxPatch((x_center-box_width/2, current_y-box_height/2), 
                               box_width, box_height,
                               boxstyle="round,pad=0.01", 
                               edgecolor='black', facecolor='white', linewidth=1.5)
    ax.add_patch(box_input)
    ax.text(x_center, current_y, '输入层\n多维特征向量', ha='center', va='center', 
            fontsize=11, fontweight='bold')
    ax.text(x_center+box_width/2+0.05, current_y, 'N×M', ha='left', va='center', 
            fontsize=10, style='italic')
    
    current_y -= (box_height/2 + arrow_length)
    
    # 箭头1
    ax.arrow(x_center, current_y+arrow_length/2, 0, -arrow_length/2, 
            head_width=0.015, head_length=0.015, fc='black', ec='black', linewidth=1.5)
    
    # 2. 第一卷积层
    current_y -= arrow_length/2
    box_conv1 = FancyBboxPatch((x_center-box_width/2, current_y-box_height/2), 
                               box_width, box_height,
                               boxstyle="round,pad=0.01", 
                               edgecolor='black', facecolor='white', linewidth=1.5)
    ax.add_patch(box_conv1)
    ax.text(x_center, current_y, '第一卷积层\nConv1D (通道16, 核3)', ha='center', va='center', 
            fontsize=10, fontweight='bold')
    ax.text(x_center+box_width/2+0.05, current_y, 'N×16', ha='left', va='center', 
            fontsize=10, style='italic')
    
    current_y -= (box_height/2 + arrow_length/2)
    
    # 箭头2
    ax.arrow(x_center, current_y+arrow_length/2, 0, -arrow_length/2, 
            head_width=0.015, head_length=0.015, fc='black', ec='black', linewidth=1.5)
    
    # 3. 批归一化层1
    current_y -= arrow_length/2
    box_bn1 = FancyBboxPatch((x_center-box_width/2, current_y-box_height/2), 
                            box_width, box_height,
                            boxstyle="round,pad=0.01", 
                            edgecolor='black', facecolor='white', linewidth=1.5)
    ax.add_patch(box_bn1)
    ax.text(x_center, current_y, '批归一化层\nBatchNorm', ha='center', va='center', 
            fontsize=10, fontweight='bold')
    
    current_y -= (box_height/2 + arrow_length/2)
    
    # 箭头3
    ax.arrow(x_center, current_y+arrow_length/2, 0, -arrow_length/2, 
            head_width=0.015, head_length=0.015, fc='black', ec='black', linewidth=1.5)
    
    # 4. ReLU激活层1
    current_y -= arrow_length/2
    box_relu1 = FancyBboxPatch((x_center-box_width/2, current_y-box_height/2), 
                              box_width, box_height,
                              boxstyle="round,pad=0.01", 
                              edgecolor='black', facecolor='white', linewidth=1.5)
    ax.add_patch(box_relu1)
    ax.text(x_center, current_y, 'ReLU激活层', ha='center', va='center', 
            fontsize=10, fontweight='bold')
    
    current_y -= (box_height/2 + arrow_length)
    
    # 箭头4
    ax.arrow(x_center, current_y+arrow_length/2, 0, -arrow_length/2, 
            head_width=0.015, head_length=0.015, fc='black', ec='black', linewidth=1.5)
    
    # 5. 第二卷积层
    current_y -= arrow_length/2
    box_conv2 = FancyBboxPatch((x_center-box_width/2, current_y-box_height/2), 
                               box_width, box_height,
                               boxstyle="round,pad=0.01", 
                               edgecolor='black', facecolor='white', linewidth=1.5)
    ax.add_patch(box_conv2)
    ax.text(x_center, current_y, '第二卷积层\nConv1D (通道32, 核3)', ha='center', va='center', 
            fontsize=10, fontweight='bold')
    ax.text(x_center+box_width/2+0.05, current_y, 'N×32', ha='left', va='center', 
            fontsize=10, style='italic')
    
    current_y -= (box_height/2 + arrow_length/2)
    
    # 箭头5
    ax.arrow(x_center, current_y+arrow_length/2, 0, -arrow_length/2, 
            head_width=0.015, head_length=0.015, fc='black', ec='black', linewidth=1.5)
    
    # 6. 批归一化层2
    current_y -= arrow_length/2
    box_bn2 = FancyBboxPatch((x_center-box_width/2, current_y-box_height/2), 
                            box_width, box_height,
                            boxstyle="round,pad=0.01", 
                            edgecolor='black', facecolor='white', linewidth=1.5)
    ax.add_patch(box_bn2)
    ax.text(x_center, current_y, '批归一化层\nBatchNorm', ha='center', va='center', 
            fontsize=10, fontweight='bold')
    
    current_y -= (box_height/2 + arrow_length/2)
    
    # 箭头6
    ax.arrow(x_center, current_y+arrow_length/2, 0, -arrow_length/2, 
            head_width=0.015, head_length=0.015, fc='black', ec='black', linewidth=1.5)
    
    # 7. ReLU激活层2
    current_y -= arrow_length/2
    box_relu2 = FancyBboxPatch((x_center-box_width/2, current_y-box_height/2), 
                              box_width, box_height,
                              boxstyle="round,pad=0.01", 
                              edgecolor='black', facecolor='white', linewidth=1.5)
    ax.add_patch(box_relu2)
    ax.text(x_center, current_y, 'ReLU激活层', ha='center', va='center', 
            fontsize=10, fontweight='bold')
    
    current_y -= (box_height/2 + arrow_length)
    
    # 箭头7
    ax.arrow(x_center, current_y+arrow_length/2, 0, -arrow_length/2, 
            head_width=0.015, head_length=0.015, fc='black', ec='black', linewidth=1.5)
    
    # 8. 扁平化层
    current_y -= arrow_length/2
    box_flatten = FancyBboxPatch((x_center-box_width/2, current_y-box_height/2), 
                                box_width, box_height,
                                boxstyle="round,pad=0.01", 
                                edgecolor='black', facecolor='white', linewidth=1.5)
    ax.add_patch(box_flatten)
    ax.text(x_center, current_y, '扁平化层\nFlatten', ha='center', va='center', 
            fontsize=10, fontweight='bold')
    
    current_y -= (box_height/2 + arrow_length)
    
    # 箭头8
    ax.arrow(x_center, current_y+arrow_length/2, 0, -arrow_length/2, 
            head_width=0.015, head_length=0.015, fc='black', ec='black', linewidth=1.5)
    
    # 9. 全连接层
    current_y -= arrow_length/2
    box_fc = FancyBboxPatch((x_center-box_width/2, current_y-box_height/2), 
                           box_width, box_height,
                           boxstyle="round,pad=0.01", 
                           edgecolor='black', facecolor='white', linewidth=1.5)
    ax.add_patch(box_fc)
    ax.text(x_center, current_y, '全连接层\nFC (64节点, Dropout=0.3)', ha='center', va='center', 
            fontsize=10, fontweight='bold')
    ax.text(x_center+box_width/2+0.05, current_y, '64', ha='left', va='center', 
            fontsize=10, style='italic')
    
    current_y -= (box_height/2 + arrow_length)
    
    # 箭头9
    ax.arrow(x_center, current_y+arrow_length/2, 0, -arrow_length/2, 
            head_width=0.015, head_length=0.015, fc='black', ec='black', linewidth=1.5)
    
    # 10. Softmax输出层
    current_y -= arrow_length/2
    box_output = FancyBboxPatch((x_center-box_width/2, current_y-box_height/2), 
                                box_width, box_height,
                                boxstyle="round,pad=0.01", 
                                edgecolor='black', facecolor='white', linewidth=1.5)
    ax.add_patch(box_output)
    ax.text(x_center, current_y, 'Softmax输出层', ha='center', va='center', 
            fontsize=10, fontweight='bold')
    ax.text(x_center+box_width/2+0.05, current_y, '3', ha='left', va='center', 
            fontsize=10, style='italic')
    
    # 输出类别标注
    output_y = current_y - box_height/2 - 0.05
    ax.text(x_center, output_y, 'I级(软土)  II级(硬岩)  III级(交错层)', 
            ha='center', va='top', fontsize=10)
    
    # 右侧参数说明
    param_x = 0.85
    param_y_start = 0.9
    param_text = [
        '网络参数：',
        '• 卷积核大小: 3',
        '• 步长: 1',
        '• 零填充: 是',
        '• 激活函数: ReLU',
        '• 优化器: Adam',
        '• 学习率: 0.001'
    ]
    for i, text in enumerate(param_text):
        ax.text(param_x, param_y_start - i*0.06, text, ha='left', va='top', 
                fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, format='png', bbox_inches='tight', facecolor='white')
    print(f"1D-CNN结构图已保存至: {save_path}")
    plt.close()

def plot_1dcnn_structure_simple(save_path='figure5_1dcnn_simple.png'):
    """
    绘制一维卷积神经网络（1D-CNN）结构示意图（简化版）
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 12))
    ax.axis('off')
    
    # 定义位置参数
    x_center = 0.5
    y_start = 0.95
    layer_height = 0.08
    arrow_length = 0.05
    box_width = 0.4
    box_height = 0.06
    
    current_y = y_start
    
    # 定义所有层
    layers = [
        ('输入层', '多维特征向量', 'N×M'),
        ('第一卷积层', 'Conv1D (通道16, 核3)', 'N×16'),
        ('批归一化层', 'BatchNorm', None),
        ('ReLU激活层', 'ReLU', None),
        ('第二卷积层', 'Conv1D (通道32, 核3)', 'N×32'),
        ('批归一化层', 'BatchNorm', None),
        ('ReLU激活层', 'ReLU', None),
        ('扁平化层', 'Flatten', None),
        ('全连接层', 'FC (64节点, Dropout=0.3)', '64'),
        ('Softmax输出层', 'Softmax', '3')
    ]
    
    for i, (layer_name, layer_desc, dim) in enumerate(layers):
        # 绘制层框
        box = FancyBboxPatch((x_center-box_width/2, current_y-box_height/2), 
                            box_width, box_height,
                            boxstyle="round,pad=0.01", 
                            edgecolor='black', facecolor='white', linewidth=1.5)
        ax.add_patch(box)
        
        # 层名称和描述
        if '\n' in layer_desc or len(layer_desc) > 15:
            text = f'{layer_name}\n{layer_desc}'
        else:
            text = f'{layer_name}\n{layer_desc}'
        ax.text(x_center, current_y, text, ha='center', va='center', 
                fontsize=10, fontweight='bold')
        
        # 维度标注
        if dim:
            ax.text(x_center+box_width/2+0.04, current_y, dim, ha='left', va='center', 
                    fontsize=9, style='italic')
        
        # 箭头（除了最后一层）
        if i < len(layers) - 1:
            current_y -= (box_height/2 + arrow_length)
            ax.arrow(x_center, current_y+arrow_length/2, 0, -arrow_length/2, 
                    head_width=0.012, head_length=0.012, fc='black', ec='black', linewidth=1.5)
            current_y -= arrow_length/2
        else:
            # 输出类别
            output_y = current_y - box_height/2 - 0.04
            ax.text(x_center, output_y, 'I级(软土)  II级(硬岩)  III级(交错层)', 
                    ha='center', va='top', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, format='png', bbox_inches='tight', facecolor='white')
    print(f"1D-CNN结构图（简化版）已保存至: {save_path}")
    plt.close()

def main():
    """主函数"""
    print("=" * 60)
    print("专利附图5：一维卷积神经网络（1D-CNN）结构示意图生成")
    print("=" * 60)
    
    print("\n1. 绘制图5：1D-CNN结构示意图（完整版）...")
    plot_1dcnn_structure('figure5_1dcnn.png')
    
    print("\n2. 绘制图5：1D-CNN结构示意图（简化版）...")
    plot_1dcnn_structure_simple('figure5_1dcnn_simple.png')
    
    print("\n" + "=" * 60)
    print("所有图表生成完成！")
    print("=" * 60)
    print("\n生成的文件：")
    print("  - figure5_1dcnn.png: 1D-CNN结构图（完整版）")
    print("  - figure5_1dcnn_simple.png: 1D-CNN结构图（简化版，推荐）")
    print("\n提示：这些图可以直接用于专利附图，或根据需要进行进一步编辑。")

if __name__ == '__main__':
    main()


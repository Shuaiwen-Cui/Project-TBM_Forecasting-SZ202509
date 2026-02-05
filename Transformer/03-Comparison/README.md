# 论文图片生成脚本使用说明

## 功能说明

本脚本用于生成论文中所需的所有对比分析图表（图3-图16）。

## 生成的图片列表

### 基于CSV数据的图片（图3-12）
- 图3：四种模型预测精度对比（R²）
- 图4：四种模型预测精度对比（MSE、MAE、RMSE）
- 图5：四种模型推理时间对比
- 图6：四种模型内存占用对比
- 图7：四种模型训练时间对比
- 图8：序列长度对预测性能的影响
- 图9：预测长度对预测性能的影响
- 图10：时间尺度组合热力图
- 图11：精度-效率权衡散点图
- 图12：模型综合性能雷达图

### 基于PKL数据的图片（图13-16）
- 图13：关键参数预测效果对比（贯入度）
- 图14：关键参数预测效果对比（推进压力）
- 图15：关键参数预测效果对比（刀盘转速）
- 图16：多参数预测误差分布箱线图

## 使用方法

### 1. 安装依赖

```bash
pip install pandas numpy matplotlib seaborn
```

### 2. 配置参数

打开 `generate_figures.py`，在文件开头修改配置参数：

- **路径配置**：数据文件路径
- **字体配置**：中文字体（宋体）、英文字体（Times New Roman）
- **图片配置**：分辨率、大小等
- **颜色配置**：各模型的颜色
- **PKL配置**：用于预测效果图的参数选择

### 3. 运行脚本

```bash
cd Transformer/03-Comparison
python generate_figures.py
```

### 4. 查看结果

生成的图片保存在 `Transformer/03-Comparison/figures/` 目录下。

## 配置参数说明

### 字体配置
- `CHINESE_FONT = 'SimSun'`：中文字体（宋体）
- `ENGLISH_FONT = 'Times New Roman'`：英文字体（Times）
- 可根据系统安装的字体调整

### 图片配置
- `DPI = 300`：图片分辨率（建议300用于论文）
- `FIG_SIZE = (10, 6)`：默认图片大小

### PKL配置
- `seq_len`：用于预测效果图的序列长度（默认60）
- `pred_len`：用于预测效果图的预测长度（默认1）
- `feature_indices`：各参数的索引位置
- `sample_range`：展示的样本范围

## 注意事项

1. 确保数据文件存在：
   - `Transformer/02-Processing/results/experiment_summary.csv`
   - `Transformer/02-Processing/results/*_results.pkl`

2. 如果系统没有安装宋体，可以修改为其他中文字体：
   - Windows: 'SimSun', 'Microsoft YaHei'
   - Mac: 'STSong', 'Arial Unicode MS'
   - Linux: 'WenQuanYi Micro Hei', 'Noto Sans CJK SC'

3. 如果PKL文件不存在，图13-16可能无法生成，脚本会跳过并给出警告。

## 自定义修改

所有可配置参数都在脚本文件开头，方便快速调整：
- 修改颜色方案
- 调整图片大小和分辨率
- 更改字体设置
- 选择不同的数据范围

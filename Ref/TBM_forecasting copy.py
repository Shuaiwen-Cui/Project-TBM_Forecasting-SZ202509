# LIBRARIES
import onnxruntime as ort
import numpy as np
import joblib
import os
import time
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from datetime import datetime
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# LOAD SCALER AND AI MODEL - SCALER FOR PREPROCESSING AND AI MODEL FOR PREDICTION
model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'AI-Model')

try:
    session = ort.InferenceSession(os.path.join(model_dir, 'model_transformer.onnx'))
    scaler = joblib.load(os.path.join(model_dir, 'minmax_scaler.pkl'))
    print("✓ 模型和归一化模块加载成功")
except FileNotFoundError as e:
    print(f"❌ 文件未找到: {e}")
    exit(1)
except Exception as e:
    print(f"❌ 加载失败: {e}")
    exit(1)

# CONFIGURATION
FEATURE_NUM = 31
PREDICTION_INPUT_LENGTH = 5
PREDICTION_OUTPUT_LENGTH = 1
VISUAL_WINDOW_LENGTH = 100
UPDATE_INTERVAL = 2  # 增加到5秒，减少闪烁

# VISUALIZATION - GROUND TRUTH AND PREDICTION
## Buffers
ground_truth_visual_buffer = np.zeros((VISUAL_WINDOW_LENGTH, FEATURE_NUM))
prediction_visual_buffer = np.zeros((VISUAL_WINDOW_LENGTH, FEATURE_NUM))
time_visual_buffer = np.zeros(VISUAL_WINDOW_LENGTH, dtype=object)  # 存储时间字符串

## Buffer Index
buffer_index = 0

## Feature Names (31个特征的中文名称)
feature_names = [
    '贯入度', '推进区间的压力（上）', '推进区间的压力（右）', '推进区间的压力（下）', '推进区间的压力（左）',
    '土舱土压（右）', '土舱土压（右下）', '土舱土压（左）', '土舱土压（左下）', 'No.16推进千斤顶速度',
    'No.4推进千斤顶速度', 'No.8推进千斤顶速度', 'No.12推进千斤顶速度', '推进油缸总推力', 'No.16推进千斤顶行程',
    'No.4推进千斤顶行程', 'No.8推进千斤顶行程', 'No.12推进千斤顶行程', '推进平均速度', '刀盘转速',
    '刀盘扭矩', 'No.1刀盘电机扭矩', 'No.2刀盘电机扭矩', 'No.3刀盘电机扭矩', 'No.4刀盘电机扭矩',
    'No.5刀盘电机扭矩', 'No.6刀盘电机扭矩', 'No.7刀盘电机扭矩', 'No.8刀盘电机扭矩', 'No.9刀盘电机扭矩',
    'No.10刀盘电机扭矩'
]

## Initialize Visualization
# 当前选择的特征索引
current_feature = 0

# 创建plotly图表
fig = go.Figure()

# 添加初始空线条
fig.add_trace(go.Scatter(
    x=[],
    y=[],
    mode='lines',
    name='Ground Truth',
    line=dict(color='blue', width=2)
))

fig.add_trace(go.Scatter(
    x=[],
    y=[],
    mode='lines',
    name='Prediction',
    line=dict(color='red', width=2, dash='dash')
))

# 更新布局
fig.update_layout(
    title=f'{feature_names[current_feature]} (特征{current_feature})',
    xaxis_title='Time Steps',
    yaxis_title='Value',
    hovermode='x unified',
    showlegend=True,
    width=1000,
    height=600,
    updatemenus=[
        dict(
            type="dropdown",
            direction="down",
            showactive=True,
            x=0.1,
            xanchor="left",
            y=1.02,
            yanchor="top",
            buttons=list([
                dict(
                    label=name,
                    method="update",
                    args=[
                        {"title": f"{name} (特征{i})"},
                        {"visible": [True, True]}
                    ]
                ) for i, name in enumerate(feature_names)
            ]),
        )
    ]
)

def update_feature(selected_feature):
    """更新选择的特征"""
    global current_feature
    current_feature = feature_names.index(selected_feature)
    update_visualization()

def update_visualization():
    """更新可视化图表 - 生成新的HTML文件，显示整个显存数据"""
    global current_feature
    
    # 始终显示整个显存的数据（100个数据点）
    valid_length = VISUAL_WINDOW_LENGTH  # 100
    gt_data = ground_truth_visual_buffer[:, current_feature]
    pred_data = prediction_visual_buffer[:, current_feature]
    
    # 计算实际有效的数据点（非零数据）
    gt_nonzero = gt_data[gt_data != 0]
    pred_nonzero = pred_data[pred_data != 0]
    actual_data_count = len(gt_nonzero)
    
    # 添加更详细的调试信息
    print(f"\n可视化调试信息:")
    print(f"  当前特征: {current_feature} ({feature_names[current_feature]})")
    print(f"  显存总长度: {valid_length}")
    print(f"  实际数据点数: {actual_data_count}")
    print(f"  Buffer Index: {buffer_index}")
    print(f"  数据更新方向: 从前面往后压 (位置0=最旧数据, 位置99=最新数据)")
    print(f"  时间步范围: 0 到 {valid_length - 1}")
    print(f"  Ground Truth 非零数据: {len(gt_nonzero)} 个")
    print(f"  Prediction 非零数据: {len(pred_nonzero)} 个")
    if len(gt_nonzero) > 0:
        print(f"  Ground Truth 数据范围: {gt_nonzero.min():.3f} 到 {gt_nonzero.max():.3f}")
    if len(pred_nonzero) > 0:
        print(f"  Prediction 数据范围: {pred_nonzero.min():.3f} 到 {pred_nonzero.max():.3f}")
    
    time_steps = np.arange(valid_length)
    
    # 创建新的图表
    fig_new = go.Figure()
    
    # 准备时间标签（用于hover显示）
    time_labels = []
    for i in range(valid_length):
        if i < len(time_visual_buffer) and time_visual_buffer[i] is not None:
            time_labels.append(f"时间: {time_visual_buffer[i]}<br>位置: {i}")
        else:
            time_labels.append(f"位置: {i}")
    
    # 添加线条
    fig_new.add_trace(go.Scatter(
        x=time_steps,
        y=gt_data,
        mode='lines',
        name='Ground Truth',
        line=dict(color='blue', width=2),
        hovertemplate='<b>Ground Truth</b><br>%{text}<br>值: %{y:.3f}<extra></extra>',
        text=time_labels
    ))
    
    fig_new.add_trace(go.Scatter(
        x=time_steps,
        y=pred_data,
        mode='lines',
        name='Prediction',
        line=dict(color='red', width=2, dash='dash'),
        hovertemplate='<b>Prediction</b><br>%{text}<br>值: %{y:.3f}<extra></extra>',
        text=time_labels
    ))
    
    # 添加当前数据位置的垂直线
    if actual_data_count > 0:
        # 由于数据从前面往后压，最新数据在位置99（或实际数据末尾）
        if actual_data_count < VISUAL_WINDOW_LENGTH:
            newest_pos = actual_data_count - 1
        else:
            newest_pos = VISUAL_WINDOW_LENGTH - 1
            
        fig_new.add_vline(
            x=newest_pos,
            line_dash="dot",
            line_color="green",
            line_width=2,
            annotation_text=f"最新数据位置: {newest_pos}",
            annotation_position="top"
        )
        
        # 添加最旧数据位置指示器
        oldest_pos = 0
        fig_new.add_vline(
            x=oldest_pos,
            line_dash="dot",
            line_color="orange",
            line_width=1,
            annotation_text=f"最旧数据位置: {oldest_pos}",
            annotation_position="bottom"
        )
    
    # 准备时间范围信息
    time_range_info = ""
    if actual_data_count > 0:
        # 找到最早和最晚的时间
        valid_times = [t for t in time_visual_buffer if t is not None and t != ""]
        if len(valid_times) > 0:
            earliest_time = valid_times[0] if len(valid_times) > 0 else "N/A"
            latest_time = valid_times[-1] if len(valid_times) > 0 else "N/A"
            time_range_info = f"<br>时间范围: {earliest_time} - {latest_time}"
    
    # 更新布局
    fig_new.update_layout(
        title=f'{feature_names[current_feature]} (特征{current_feature}) - 当前时间: {datetime.now().strftime("%H:%M:%S")}<br>显示整个显存数据 ({actual_data_count}/{valid_length} 个有效数据点){time_range_info}',
        xaxis_title='Time Steps (显存位置 0-99)',
        yaxis_title='Value',
        hovermode='x unified',
        showlegend=True,
        width=1200,
        height=700,
        updatemenus=[
            dict(
                type="dropdown",
                direction="down",
                showactive=True,
                x=0.1,
                xanchor="left",
                y=1.02,
                yanchor="top",
                buttons=list([
                    dict(
                        label=name,
                        method="update",
                        args=[
                            {
                                "title": f"{name} (特征{i}) - 当前时间: {datetime.now().strftime('%H:%M:%S')}<br>显示整个显存数据 ({actual_data_count}/{valid_length} 个有效数据点){time_range_info}"
                            },
                            {
                                "y": [ground_truth_visual_buffer[:, i].tolist(), prediction_visual_buffer[:, i].tolist()],
                                "x": [list(range(valid_length)), list(range(valid_length))],
                                "text": [time_labels, time_labels]
                            }
                        ]
                    ) for i, name in enumerate(feature_names)
                ]),
            ),
            dict(
                type="buttons",
                direction="left",
                showactive=True,
                x=0.9,
                xanchor="right",
                y=1.02,
                yanchor="top",
                buttons=list([
                    dict(
                        label="手动刷新",
                        method="restyle",
                        args=[{"visible": [True, True]}]
                    ),
                    dict(
                        label="暂停自动刷新",
                        method="restyle",
                        args=[{"visible": [True, True]}]
                    )
                ]),
            )
        ]
    )
    
    # 自动调整Y轴范围
    all_data = np.concatenate([gt_data, pred_data])
    if len(all_data) > 0 and valid_length > 0:
        if np.ptp(all_data) > 0:  # 避免除零错误
            margin = np.ptp(all_data) * 0.1
            y_min = np.min(all_data) - margin
            y_max = np.max(all_data) + margin
            fig_new.update_yaxes(range=[y_min, y_max])
        else:
            center = np.mean(all_data)
            fig_new.update_yaxes(range=[center - 1, center + 1])
    
    # 更新X轴范围 - 显示完整的显存范围
    fig_new.update_xaxes(
        range=[0, valid_length - 1],  # 0 到 99
        title="Time Steps (显存位置 0-99)",
        tickmode='linear',
        tick0=0,
        dtick=10,  # 每10个单位显示一个刻度
        showgrid=True,
        gridwidth=1,
        gridcolor='lightgray'
    )
    
    # 保存为HTML文件并打开
    html_file = "tbm_prediction.html"
    
    # 添加自动刷新JavaScript代码
    auto_refresh_js = f"""
    <script>
    // 每{UPDATE_INTERVAL}秒自动刷新页面，添加平滑过渡
    let refreshCount = 0;
    const maxRefreshes = 1000; // 最大刷新次数，防止无限刷新
    
    function refreshPage() {{
        refreshCount++;
        if (refreshCount >= maxRefreshes) {{
            console.log('达到最大刷新次数，停止自动刷新');
            return;
        }}
        
        // 添加淡出效果
        document.body.style.transition = 'opacity 0.5s ease-out';
        document.body.style.opacity = '0.7';
        
        setTimeout(function() {{
            location.reload();
        }}, 500); // 0.5秒后刷新
    }}
    
    // 每{UPDATE_INTERVAL}秒执行一次刷新
    setInterval(refreshPage, {UPDATE_INTERVAL * 1000});
    
    // 页面加载完成后的淡入效果
    window.addEventListener('load', function() {{
        document.body.style.transition = 'opacity 0.5s ease-in';
        document.body.style.opacity = '1';
        
        // 修复dropdown按钮功能
        console.log('页面加载完成，dropdown按钮已就绪');
    }});
    
    // 添加特征切换功能
    function switchFeature(featureIndex) {{
        console.log('切换到特征:', featureIndex);
        // 这里可以添加AJAX请求来获取新数据，或者重新加载页面
        // 目前使用页面刷新来更新数据
        location.reload();
    }}
    </script>
    """
    
    # 写入HTML文件
    html_content = fig_new.to_html(include_plotlyjs=True)
    # 在</body>标签前插入自动刷新代码
    html_content = html_content.replace('</body>', auto_refresh_js + '</body>')
    
    with open(html_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    # 只在第一次显示图表
    if not hasattr(update_visualization, 'shown'):
        try:
            import webbrowser
            webbrowser.open(f"file://{os.path.abspath(html_file)}")
            update_visualization.shown = True
            print("✓ 实时图表已生成")
            print(f"  - 图表文件: {html_file}")
            print("  - 浏览器会自动打开图表")
            print(f"  - 图表每{UPDATE_INTERVAL}秒自动更新")
            print("  - 使用图表上方的下拉菜单选择不同特征")
        except:
            print("✓ 图表已保存为HTML文件")
            print(f"  - 请手动打开: {html_file}")
            update_visualization.shown = True

print("使用Plotly交互式图表模式")

# PREDICTION INPUT & OUTPUT
## Input Buffer - PREDICTION_INPUT_LENGTH steps
prediction_input_buffer = np.zeros((1, PREDICTION_INPUT_LENGTH, FEATURE_NUM))

## Output Buffer - PREDICTION_OUTPUT_LENGTH step
prediction_output_buffer = np.zeros((1, FEATURE_NUM))

# LOOP FOR CONTINUOUS PREDICTION
try:
    while True:
        # 生成新的随机数据（模拟实时传感器数据）
        new_data = np.zeros((1, 1, FEATURE_NUM))
        for i in range(FEATURE_NUM):
            min_val = scaler.data_min_[i]
            max_val = scaler.data_max_[i]
            new_data[0, 0, i] = np.random.uniform(min_val, max_val, 1)
        
        # 滑动窗口：向左移动，压入新数据
        prediction_input_buffer[0, :-1, :] = prediction_input_buffer[0, 1:, :]  # 左移
        prediction_input_buffer[0, -1, :] = new_data[0, 0, :]  # 压入新数据
        
        # 获取当前时间（24小时制，精确到秒）
        current_time_str = datetime.now().strftime('%H:%M:%S')
        
        # 更新ground_truth_visual_buffer（从前面往后压）
        if buffer_index < VISUAL_WINDOW_LENGTH:
            # 如果还没填满，直接填充到当前位置
            ground_truth_visual_buffer[buffer_index, :] = new_data[0, 0, :]
            time_visual_buffer[buffer_index] = current_time_str
        else:
            # 如果已填满，从前面往后压（左移）
            ground_truth_visual_buffer[:-1, :] = ground_truth_visual_buffer[1:, :]  # 左移
            ground_truth_visual_buffer[-1, :] = new_data[0, 0, :]  # 压入新数据到最后面
            time_visual_buffer[:-1] = time_visual_buffer[1:]  # 时间buffer左移
            time_visual_buffer[-1] = current_time_str  # 压入新时间到最后面
        
        # Continuous Prediction Logic
        print(f"执行预测 - 时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"输入数据形状: {prediction_input_buffer.shape}")
        
        # 模型预测（参考forecasting.py）
        # 步骤1: 数据标准化
        input_data_scaled = scaler.transform(prediction_input_buffer.reshape(-1, FEATURE_NUM)).reshape(1, PREDICTION_INPUT_LENGTH, FEATURE_NUM).astype(np.float32)
        
        # 步骤2: 模型推理
        output_scaled = session.run(None, {session.get_inputs()[0].name: input_data_scaled})
        
        # 步骤3: 反标准化
        prediction_output_buffer[0, :] = scaler.inverse_transform(output_scaled[0])
        
        # 更新prediction_visual_buffer（从前面往后压）
        if buffer_index < VISUAL_WINDOW_LENGTH:
            # 如果还没填满，直接填充到当前位置
            prediction_visual_buffer[buffer_index, :] = prediction_output_buffer[0, :]
        else:
            # 如果已填满，从前面往后压（左移）
            prediction_visual_buffer[:-1, :] = prediction_visual_buffer[1:, :]  # 左移
            prediction_visual_buffer[-1, :] = prediction_output_buffer[0, :]  # 压入预测结果到最后面
        
        # 更新buffer_index（在数据更新后）
        buffer_index = (buffer_index + 1) % VISUAL_WINDOW_LENGTH
        
        print(f"预测结果形状: {prediction_output_buffer.shape}")
        print(f"Ground truth buffer索引: {buffer_index}")
        
        # 添加显存调试信息
        if buffer_index % 5 == 0:
            print(f"\n显存调试信息:")
            print(f"  Ground Truth Buffer 最新数据: {ground_truth_visual_buffer[-1, :5]}")  # 显示前5个特征
            print(f"  Prediction Buffer 最新数据: {prediction_visual_buffer[-1, :5]}")  # 显示前5个特征
            print(f"  Buffer Index: {buffer_index}")
            print(f"  Ground Truth Buffer 形状: {ground_truth_visual_buffer.shape}")
            print(f"  Prediction Buffer 形状: {prediction_visual_buffer.shape}")
        
        # 在终端中输出当前预测结果
        print("\n" + "="*80)
        print(f"预测结果 - 时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"显存时间范围: {time_visual_buffer[0] if time_visual_buffer[0] is not None else 'N/A'} - {time_visual_buffer[-1] if time_visual_buffer[-1] is not None else 'N/A'}")
        print(f"Buffer Index: {buffer_index} (位置0=最旧, 位置99=最新)")
        print("="*80)
        
        # 显示全部31个特征的预测结果
        for i in range(FEATURE_NUM):
            gt_val = ground_truth_visual_buffer[-1, i]  # 使用最新的数据
            pred_val = prediction_visual_buffer[-1, i]  # 使用最新的数据
            error = abs(pred_val - gt_val)
            error_pct = (error / gt_val * 100) if gt_val != 0 else 0
            
            print(f"特征{i:2d} ({feature_names[i]:<20}): "
                  f"真实值={gt_val:8.3f}, 预测值={pred_val:8.3f}, "
                  f"误差={error:6.3f} ({error_pct:5.1f}%)")
        
        print("="*80)
        
        # 更新可视化（每UPDATE_INTERVAL秒更新一次HTML文件）
        update_visualization()
        print(f"✓ 图表已更新 - 文件: tbm_prediction.html")
        
        # 计算到下一个UPDATE_INTERVAL秒间隔的等待时间
        current_time = time.time()
        next_interval = ((int(current_time) // UPDATE_INTERVAL) + 1) * UPDATE_INTERVAL
        sleep_time = next_interval - current_time
        time.sleep(sleep_time)
        
except KeyboardInterrupt:
    print("\n Program Interrupted by User")
    exit(0)

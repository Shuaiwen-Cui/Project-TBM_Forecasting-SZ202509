#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TBM盾构机预测系统
============================

功能: 基于历史5步数据预测下一步的31个特征值
时间逻辑: 输入t-4到t，预测t+1
"""

import onnxruntime as ort
import numpy as np
import joblib
import os
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# 配置参数
# =============================================================================
FEATURE_NUM = 31
INPUT_LENGTH = 5
UPDATE_INTERVAL = 2

# 特征名称
FEATURE_NAMES = [
    '贯入度',                           # 0
    '推进区间的压力（上）',                # 1
    '推进区间的压力（右）',                # 2
    '推进区间的压力（下）',                # 3
    '推进区间的压力（左）',                # 4
    '土舱土压（右）',                    # 5
    '土舱土压（右下）',                   # 6
    '土舱土压（左）',                    # 7
    '土舱土压（左下）',                   # 8
    'No.16推进千斤顶速度',               # 9
    'No.4推进千斤顶速度',                # 10
    'No.8推进千斤顶速度',                # 11
    'No.12推进千斤顶速度',               # 12
    '推进油缸总推力',                    # 13
    'No.16推进千斤顶行程',               # 14
    'No.4推进千斤顶行程',                # 15
    'No.8推进千斤顶行程',                # 16
    'No.12推进千斤顶行程',               # 17
    '推进平均速度',                      # 18
    '刀盘转速',                         # 19
    '刀盘扭矩',                         # 20
    'No.1刀盘电机扭矩',                 # 21
    'No.2刀盘电机扭矩',                 # 22
    'No.3刀盘电机扭矩',                 # 23
    'No.4刀盘电机扭矩',                 # 24
    'No.5刀盘电机扭矩',                 # 25
    'No.6刀盘电机扭矩',                 # 26
    'No.7刀盘电机扭矩',                 # 27
    'No.8刀盘电机扭矩',                 # 28
    'No.9刀盘电机扭矩',                 # 29
    'No.10刀盘电机扭矩'                 # 30
]


# =============================================================================
# 模型管理
# =============================================================================
def load_model():
    """加载模型和归一化器"""
    model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'AI-Model')
    
    try:
        session = ort.InferenceSession(os.path.join(model_dir, 'model_transformer.onnx'))
        scaler = joblib.load(os.path.join(model_dir, 'minmax_scaler.pkl'))
        print("✓ 模型加载成功")
        return session, scaler
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        exit(1)


def predict(session, scaler, input_data):
    """执行预测"""
    # 标准化
    input_scaled = scaler.transform(input_data.reshape(-1, FEATURE_NUM)).reshape(1, INPUT_LENGTH, FEATURE_NUM).astype(np.float32)
    
    # 推理
    output_scaled = session.run(None, {session.get_inputs()[0].name: input_scaled})
    
    # 反标准化
    prediction = scaler.inverse_transform(output_scaled[0])
    
    return prediction[0]


# =============================================================================
# 数据管理
# =============================================================================
def generate_data(scaler):
    """生成模拟数据"""
    data = np.zeros(FEATURE_NUM)
    for i in range(FEATURE_NUM):
        min_val = scaler.data_min_[i]
        max_val = scaler.data_max_[i]
        data[i] = np.random.uniform(min_val, max_val)
    return data


def update_buffer(buffer, new_data):
    """更新滑动窗口缓冲区"""
    buffer[:-1] = buffer[1:]  # 左移
    buffer[-1] = new_data     # 压入新数据
    return buffer


# =============================================================================
# 结果输出
# =============================================================================
def print_prediction(pred_values, step):
    """打印预测结果"""
    print(f"\n步骤 {step} - 时间: {datetime.now().strftime('%H:%M:%S')}")
    print(f"预测值 (基于t-4到t预测t+1):")
    
    for i in range(FEATURE_NUM):
        pred = pred_values[i]
        print(f"特征{i:2d} ({FEATURE_NAMES[i]:<20}): {pred:8.3f}")
    
    print("-" * 60)


# =============================================================================
# 主程序
# =============================================================================
def main():
    """主程序"""
    print("TBM盾构机预测系统启动")
    print(f"特征数量: {FEATURE_NUM}, 输入长度: {INPUT_LENGTH}, 更新间隔: {UPDATE_INTERVAL}秒")
    
    # 加载模型
    session, scaler = load_model()
    
    # 初始化缓冲区
    input_buffer = np.zeros((INPUT_LENGTH, FEATURE_NUM))
    history = []
    step = 0
    
    try:
        while True:
            # 生成新数据
            new_data = generate_data(scaler)
            history.append(new_data.copy())
            
            # 更新输入缓冲区
            input_buffer = update_buffer(input_buffer, new_data)
            step += 1
            
            # 检查是否可以预测
            if step >= INPUT_LENGTH:
                # 执行预测
                prediction = predict(session, scaler, input_buffer)
                
                # 输出预测结果
                print_prediction(prediction, step - INPUT_LENGTH + 1)
            else:
                print(f"步骤 {step}: 收集数据中... (需要 {INPUT_LENGTH} 步)")
            
            # 等待下一个间隔
            time.sleep(UPDATE_INTERVAL)
            
    except KeyboardInterrupt:
        print(f"\n程序结束 - 总步数: {step}")


if __name__ == "__main__":
    main()

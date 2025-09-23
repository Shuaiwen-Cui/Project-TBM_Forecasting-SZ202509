#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TBM盾构机预测系统 (main.py)
==========================

系统功能:
- 基于历史5步数据预测下一步的31个特征值
- 时间逻辑: 输入t-4到t，预测t+1
- 数据源: 随机数据 -> 真实数据(API)

系统架构:
- 主程序控制模块: 系统初始化和运行控制
- 数据管理模块: 数据生成和滑动窗口管理
- 预测执行模块: 调用predict.py进行模型预测
- 结果展示模块: 预测结果格式化输出
"""

# =============================================================================
# 导入依赖库
# =============================================================================
import numpy as np
import time
import random
from datetime import datetime
from predict import ModelPredictor  # 导入预测模块


# =============================================================================
# 系统配置模块
# =============================================================================
# 模型参数配置
FEATURE_NUM = 31        # 特征数量
INPUT_LENGTH = 5        # 输入时间步长度
UPDATE_INTERVAL = 2     # 更新间隔(秒)

# 特征名称定义
FEATURE_NAMES = [
    '贯入度', '推进区间的压力（上）', '推进区间的压力（右）', '推进区间的压力（下）', '推进区间的压力（左）',
    '土舱土压（右）', '土舱土压（右下）', '土舱土压（左）', '土舱土压（左下）',
    'No.16推进千斤顶速度', 'No.4推进千斤顶速度', 'No.8推进千斤顶速度', 'No.12推进千斤顶速度',
    '推进油缸总推力', 'No.16推进千斤顶行程', 'No.4推进千斤顶行程', 'No.8推进千斤顶行程', 'No.12推进千斤顶行程',
    '推进平均速度', '刀盘转速', '刀盘扭矩',
    'No.1刀盘电机扭矩', 'No.2刀盘电机扭矩', 'No.3刀盘电机扭矩', 'No.4刀盘电机扭矩', 'No.5刀盘电机扭矩',
    'No.6刀盘电机扭矩', 'No.7刀盘电机扭矩', 'No.8刀盘电机扭矩', 'No.9刀盘电机扭矩', 'No.10刀盘电机扭矩'
]

# =============================================================================
# TBM预测系统主类
# =============================================================================
class TBMPredictor:
    """
    TBM盾构机预测系统主类
    
    职责:
    1. 系统初始化和运行控制
    2. 数据生成和滑动窗口管理
    3. 预测执行和结果展示
    4. 主程序循环控制
    """
    
    def __init__(self):
        """初始化预测系统"""
        self.model = ModelPredictor()                    # 预测模块实例
        self.buffer = np.zeros((INPUT_LENGTH, FEATURE_NUM))  # 滑动窗口缓冲区
        self.step_count = 0                              # 步数计数器
        self.last_data = None                            # 上次生成的数据
    
    # =========================================================================
    # 系统初始化模块
    # =========================================================================
    def load_model(self):
        """
        加载预测模型
        
        Returns:
            bool: 加载成功返回True，失败返回False
        """
        return self.model.load_model()
    
    # =========================================================================
    # 数据管理模块
    # =========================================================================
    def generate_data(self):
        """
        生成模拟数据
        
        数据生成策略:
        1. 如果模型未加载，生成随机数据
        2. 如果模型已加载，基于数据范围生成
        3. 如果有历史数据，基于历史数据添加变化
        
        Returns:
            np.ndarray: 生成的31维特征数据
        """
        # 模型未加载时生成随机数据
        if not self.model.is_loaded:
            return np.random.rand(FEATURE_NUM) * 100
        
        # 获取数据范围
        data_min, data_max = self.model.get_data_range()
        data = np.zeros(FEATURE_NUM)
        
        for i in range(FEATURE_NUM):
            min_val = data_min[i]
            max_val = data_max[i]
            
            if self.last_data is not None:
                # 基于历史数据生成（添加±10%变化）
                base_value = self.last_data[i]
                variation = base_value * 0.1 * (random.random() - 0.5) * 2
                data[i] = max(min_val, min(max_val, base_value + variation))
            else:
                # 首次生成，使用随机值
                data[i] = random.uniform(min_val, max_val)
        
        self.last_data = data.copy()
        return data
    
    def update_buffer(self, new_data):
        """
        更新滑动窗口缓冲区
        
        Args:
            new_data (np.ndarray): 新的数据点
        """
        # 左移数据
        self.buffer[:-1] = self.buffer[1:]
        # 添加新数据
        self.buffer[-1] = new_data
        self.step_count += 1
    
    # =========================================================================
    # 预测执行模块
    # =========================================================================
    def predict(self, input_data):
        """
        执行预测
        
        Args:
            input_data (np.ndarray): 输入数据
            
        Returns:
            np.ndarray: 预测结果
        """
        return self.model.predict(input_data)
    
    # =========================================================================
    # 结果展示模块
    # =========================================================================
    def print_prediction(self, pred_values, step):
        """
        打印预测结果
        
        Args:
            pred_values (np.ndarray): 预测值数组
            step (int): 预测步数
        """
        print(f"\n步骤 {step} - 时间: {datetime.now().strftime('%H:%M:%S')}")
        print("预测结果 (基于t-4到t预测t+1):")
        print("=" * 80)
        
        for i in range(FEATURE_NUM):
            print(f"特征{(i+1):2d} {FEATURE_NAMES[i]:<20}: {pred_values[i]:8.3f}")
        print("=" * 80)
    
    # =========================================================================
    # 主程序控制模块
    # =========================================================================
    def run(self):
        """
        主运行循环
        
        运行流程:
        1. 加载模型
        2. 进入主循环
        3. 生成数据 -> 更新缓冲区 -> 执行预测 -> 输出结果
        4. 处理中断信号
        """
        # 步骤1: 加载模型
        if not self.load_model():
            return
        
        # 步骤2: 显示系统信息
        print("TBM盾构机预测系统启动")
        print(f"特征数量: {FEATURE_NUM}, 输入长度: {INPUT_LENGTH}, 更新间隔: {UPDATE_INTERVAL}秒")
        
        try:
            # 步骤3: 主循环
            while True:
                # 3.1 生成新数据
                new_data = self.generate_data()
                
                # 3.2 更新滑动窗口
                self.update_buffer(new_data)
                
                # 3.3 检查是否可以预测
                if self.step_count >= INPUT_LENGTH:
                    # 执行预测
                    prediction = self.predict(self.buffer)
                    self.print_prediction(prediction, self.step_count - INPUT_LENGTH + 1)
                else:
                    # 显示数据收集状态
                    print(f"步骤 {self.step_count}: 收集数据中... (需要 {INPUT_LENGTH} 步)")
                
                # 3.4 等待下一个更新间隔
                time.sleep(UPDATE_INTERVAL)
                
        except KeyboardInterrupt:
            # 步骤4: 处理中断信号
            print(f"\n程序结束 - 总步数: {self.step_count}")


# =============================================================================
# 程序入口
# =============================================================================
if __name__ == "__main__":
    # 创建预测系统实例并运行
    predictor = TBMPredictor()
    predictor.run()
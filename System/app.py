#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TBM盾构机关键掘进参数实时预测系统 - 后端API服务器

提供RESTful API接口，为前端提供实时数据
"""

import os
import sys
import json
import time
import random
import numpy as np
from datetime import datetime, timedelta
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import threading
import queue

# 添加父目录到Python路径，以便导入System-API模块
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'System-API'))

try:
    from api_client import TBMAPIClient
    from predict import ModelPredictor
    API_AVAILABLE = True
except ImportError as e:
    print(f"警告: 无法导入API客户端或预测模块: {e}")
    API_AVAILABLE = False

# Flask应用配置
app = Flask(__name__)
CORS(app)  # 允许跨域请求

# 全局变量
api_client = None
model_predictor = None
data_buffer = queue.Queue(maxsize=10)
last_data = None
is_running = False

# 特征配置
FEATURE_NAMES = [
    '贯入度', '推进区间的压力（上）', '推进区间的压力（右）', '推进区间的压力（下）', '推进区间的压力（左）',
    '土舱土压（右）', '土舱土压（右下）', '土舱土压（左）', '土舱土压（左下）',
    'No.16推进千斤顶速度', 'No.4推进千斤顶速度', 'No.8推进千斤顶速度', 'No.12推进千斤顶速度',
    '推进油缸总推力', 'No.16推进千斤顶行程', 'No.4推进千斤顶行程', 'No.8推进千斤顶行程', 'No.12推进千斤顶行程',
    '推进平均速度', '刀盘转速', '刀盘扭矩',
    'No.1刀盘电机扭矩', 'No.2刀盘电机扭矩', 'No.3刀盘电机扭矩', 'No.4刀盘电机扭矩', 'No.5刀盘电机扭矩',
    'No.6刀盘电机扭矩', 'No.7刀盘电机扭矩', 'No.8刀盘电机扭矩', 'No.9刀盘电机扭矩', 'No.10刀盘电机扭矩'
]

FEATURE_UNITS = [
    'MPa', 'MPa', 'MPa', 'MPa', 'MPa',  # 贯入度, 推进压力
    'MPa', 'MPa', 'MPa', 'MPa',  # 土舱土压
    'mm/min', 'mm/min', 'mm/min', 'mm/min',  # 推进千斤顶速度
    'kN',  # 推进油缸总推力
    'mm', 'mm', 'mm', 'mm',  # 推进千斤顶行程
    'mm/min', 'r/min', 'kN·m',  # 推进平均速度, 刀盘转速, 刀盘扭矩
    '%', '%', '%', '%', '%', '%', '%', '%', '%', '%'  # 刀盘电机扭矩
]

class DataGenerator:
    """数据生成器 - 负责获取和生成TBM数据"""
    
    def __init__(self):
        self.api_client = None
        self.model_predictor = None
        self.last_prediction = None
        self.data_mode = 4  # 默认使用预测值填充模式
        self.last_values = None  # 用于重复数据检测
        
        # 滑动窗口缓冲区 (5步历史数据)
        self.buffer = np.zeros((5, 31))  # 5个时间步，31个特征
        self.step_count = 0
        self.last_api_data = None  # 用于检测API数据是否变化
        self.buffer_initialized = False  # 缓冲区是否已初始化
        
        # 定时配置
        self.DATA_FETCH_SECOND = 10  # 每分钟的第10秒拉取数据
        self.last_fetch_time = None  # 记录上次拉取时间，避免重复拉取
        
        # 数据变化检测
        self.last_fetched_data = None  # 存储上次拉取的数据，用于变化检测
        
        # 盾构机状态
        self.tbm_status = 'rest'  # 'active' 或 'rest'
        
        # 初始化API客户端和模型
        self._initialize_components()
    
    def _initialize_components(self):
        """初始化API客户端和预测模型"""
        global API_AVAILABLE
        
        if API_AVAILABLE:
            try:
                self.api_client = TBMAPIClient()
                print("✅ API客户端初始化成功")
            except Exception as e:
                print(f"❌ API客户端初始化失败: {e}")
                self.api_client = None
            
            try:
                self.model_predictor = ModelPredictor()
                if self.model_predictor.load_model():
                    print("✅ 预测模型加载成功")
                else:
                    print("❌ 预测模型加载失败")
                    self.model_predictor = None
            except Exception as e:
                print(f"❌ 预测模型初始化失败: {e}")
                self.model_predictor = None
        else:
            print("⚠️  API和模型模块不可用，将使用模拟数据")
    
    def initialize_buffer(self):
        """
        初始化缓冲区 - 拉取5分钟历史数据填充缓冲区
        
        Returns:
            bool: 初始化成功返回True，失败返回False
        """
        if self.buffer_initialized:
            print("✅ 缓冲区已初始化，跳过")
            return True
        
        print("🔄 开始初始化缓冲区...")
        print("=" * 60)
        
        if self.api_client and self.model_predictor:
            # 获取5分钟历史数据
            historical_data = self.api_client.get_historical_data(minutes_back=5)
            
            if not historical_data:
                print("❌ 无法获取历史数据，使用模拟数据初始化")
                # 使用模拟数据初始化
                for i in range(5):
                    simulated_data = self._generate_simulated_data()
                    self.buffer[i] = simulated_data
                    self.step_count += 1
                    print(f"   步骤 {i+1}: 使用模拟数据")
            else:
                # 使用历史数据初始化
                print(f"📊 使用 {len(historical_data)} 条历史记录初始化缓冲区")
                
                # 选择最近的5条记录（如果历史数据超过5条）
                if len(historical_data) >= 5:
                    selected_data = historical_data[-5:]
                else:
                    # 如果历史数据不足5条，用模拟数据补充
                    selected_data = historical_data.copy()
                    while len(selected_data) < 5:
                        simulated_data = self._generate_simulated_data()
                        selected_data.append({
                            'features': simulated_data,
                            'timestamp': 'simulated',
                            'record_id': f'sim_{len(selected_data)}',
                            'valid_count': 31
                        })
                
                # 填充缓冲区
                for i, data_record in enumerate(selected_data):
                    features = data_record['features']
                    timestamp = data_record['timestamp']
                    record_id = data_record['record_id']
                    valid_count = data_record['valid_count']
                    
                    # 填充缺失数据
                    filled_features = self._fill_missing_data(features)
                    self.buffer[i] = filled_features
                    self.step_count += 1
                    
                    print(f"   步骤 {i+1}: ID={record_id}, 时间={timestamp}, 有效特征={valid_count}/31")
        else:
            # API不可用，使用模拟数据初始化
            print("❌ API不可用，使用模拟数据初始化")
            for i in range(5):
                simulated_data = self._generate_simulated_data()
                self.buffer[i] = simulated_data
                self.step_count += 1
                print(f"   步骤 {i+1}: 使用模拟数据")
        
        self.buffer_initialized = True
        print("=" * 60)
        print(f"✅ 缓冲区初始化完成！步数: {self.step_count}/5")
        print("=" * 60)
        return True
    
    def _generate_simulated_data(self):
        """生成模拟数据"""
        data = []
        for i in range(31):
            if i < 5:  # 贯入度和推进压力 (MPa)
                data.append(random.uniform(0.5, 3.0))
            elif i < 9:  # 土舱土压 (MPa)
                data.append(random.uniform(0.1, 0.8))
            elif i < 13:  # 推进千斤顶速度 (mm/min)
                data.append(random.uniform(10, 50))
            elif i == 13:  # 推进油缸总推力 (kN)
                data.append(random.uniform(5000, 15000))
            elif i < 18:  # 推进千斤顶行程 (mm)
                data.append(random.uniform(100, 2000))
            elif i == 18:  # 推进平均速度 (mm/min)
                data.append(random.uniform(20, 80))
            elif i == 19:  # 刀盘转速 (r/min)
                data.append(random.uniform(0.5, 2.5))
            elif i == 20:  # 刀盘扭矩 (kN·m)
                data.append(random.uniform(1000, 5000))
            else:  # 刀盘电机扭矩 (%)
                data.append(random.uniform(20, 100))
        return np.array(data)
    
    def get_latest_data(self):
        """获取最新的TBM数据 - 实现与main.py相同的预测逻辑，每分钟第10秒更新"""
        try:
            # 0. 如果缓冲区未初始化，先初始化
            if not self.buffer_initialized:
                print("🔄 首次调用，初始化缓冲区...")
                if not self.initialize_buffer():
                    print("❌ 缓冲区初始化失败")
                    return {
                        'current_values': [None] * 31,
                        'current_sources': ['simulated'] * 31,
                        'prediction_values': None,
                        'step_count': 0,
                        'buffer_ready': False
                    }
            
            # 1. 获取当前时刻(t)的API数据
            api_result = self._get_current_api_data()
            
            # 检查是否到了拉取时间
            if api_result is None:
                # 不是拉取时间，返回当前缓存的数据
                current_time = datetime.now()
                print(f"⏰ 时间: {current_time.strftime('%H:%M:%S')} - 非拉取时间，返回缓存数据")
                
                # 返回当前缓冲区的最新数据
                if self.step_count > 0:
                    current_data = self.buffer[-1]  # 最新的数据（滑动窗口的最后一个位置）
                    data_sources = ['cached'] * 31  # 标记为缓存数据
                else:
                    current_data = np.full(31, None)
                    data_sources = ['simulated'] * 31
                
                # 判定盾构机状态
                tbm_status = self._determine_tbm_status(current_data)
                
                # 根据盾构机状态决定预测值
                prediction = None
                if self.step_count >= 5:
                    if tbm_status == 'active':
                        # 掘进中：使用AI预测结果
                        try:
                            prediction = self._predict_next_step()
                            print(f"🤖 使用AI预测结果 (掘进中)")
                        except Exception as e:
                            print(f"⚠️  AI预测执行失败: {e}")
                            prediction = None
                    else:
                        # 休息中：使用智能填充数据
                        prediction = self._generate_smart_fill_data(current_data)
                        print(f"🧠 使用智能填充数据 (休息中)")
                
                return {
                    'current_values': current_data.tolist() if isinstance(current_data, np.ndarray) else current_data,
                    'current_sources': data_sources,
                    'prediction_values': prediction.tolist() if isinstance(prediction, np.ndarray) else prediction,
                    'step_count': self.step_count,
                    'buffer_ready': self.step_count >= 5,
                    'tbm_status': tbm_status
                }
            
            # 处理API结果
            if isinstance(api_result, tuple):
                current_data, data_sources = api_result
            else:
                # API不可用模式，生成模拟数据
                current_data = api_result
                data_sources = ['simulated'] * 31
            
            # 确保current_data是31个元素的数组
            if not isinstance(current_data, (list, np.ndarray)) or len(current_data) != 31:
                print(f"⚠️  当前数据格式错误，使用默认值: {type(current_data)}, 长度: {len(current_data) if hasattr(current_data, '__len__') else 'N/A'}")
                current_data = np.full(31, None)
                data_sources = ['simulated'] * 31
            
            # 2. 更新滑动窗口缓冲区（挤掉最早的数据）
            self._update_buffer(current_data)
            
            # 3. 判定盾构机状态
            tbm_status = self._determine_tbm_status(current_data)
            
            # 4. 根据盾构机状态决定预测值
            prediction = None
            if self.step_count >= 5:
                if tbm_status == 'active':
                    # 掘进中：使用AI预测结果
                    try:
                        prediction = self._predict_next_step()
                        print(f"🤖 使用AI预测结果 (掘进中)")
                    except Exception as e:
                        print(f"⚠️  AI预测执行失败: {e}")
                        prediction = None
                else:
                    # 休息中：使用智能填充数据
                    prediction = self._generate_smart_fill_data(current_data)
                    print(f"🧠 使用智能填充数据 (休息中)")
            
            # 5. 返回当前值和预测值
            return {
                'current_values': current_data.tolist() if isinstance(current_data, np.ndarray) else current_data,
                'current_sources': data_sources,
                'prediction_values': prediction.tolist() if isinstance(prediction, np.ndarray) else prediction,
                'step_count': self.step_count,
                'buffer_ready': self.step_count >= 5,
                'tbm_status': tbm_status
            }
            
        except Exception as e:
            print(f"❌ 数据获取失败: {e}")
            # 返回安全的默认值
            return {
                'current_values': [None] * 31,
                'current_sources': ['simulated'] * 31,
                'prediction_values': None,
                'step_count': self.step_count,
                'buffer_ready': False
            }
    
    def _get_real_data(self):
        """获取真实API数据"""
        try:
            # 从API获取数据
            api_data = self.api_client.get_latest_features()
            
            # 检查数据是否有效
            if api_data is None or len(api_data) != 31:
                print("⚠️  API数据无效，使用预测值填充")
                return self._fill_with_predictions()
            
            # 填充缺失数据
            filled_data = self._fill_missing_data(api_data)
            
            # 更新预测值
            if self.model_predictor and self.model_predictor.is_loaded:
                try:
                    # 这里可以添加预测逻辑
                    pass
                except Exception as e:
                    print(f"预测执行失败: {e}")
            
            return filled_data
            
        except Exception as e:
            print(f"❌ 获取真实数据失败: {e}")
            return self._fill_with_predictions()
    
    def _fill_missing_data(self, api_data, fill_mode=3):
        """填充缺失数据 - 与main.py逻辑一致"""
        filled_data = api_data.copy() if hasattr(api_data, 'copy') else api_data[:]
        
        for i in range(31):
            if api_data[i] is None:  # 数据缺失
                if fill_mode == 1:  # 0填充
                    filled_data[i] = 0.0
                elif fill_mode == 2:  # 随机填充
                    filled_data[i] = self._generate_realistic_value(i)
                elif fill_mode == 3:  # 预测值填充
                    if self.last_prediction is not None:
                        filled_data[i] = self.last_prediction[i]
                    else:
                        filled_data[i] = self._generate_realistic_value(i)
                elif fill_mode == 4:  # 预测值填充
                    if self.last_prediction is not None:
                        filled_data[i] = self.last_prediction[i]
                    else:
                        filled_data[i] = self._generate_realistic_value(i)
                else:
                    filled_data[i] = self._generate_realistic_value(i)
        
        return filled_data
    
    def _fill_with_predictions(self):
        """完全使用预测值填充"""
        if self.last_prediction is not None:
            return [{'value': pred, 'predicted': True, 'original': None} 
                   for pred in self.last_prediction]
        else:
            return [{'value': self._generate_realistic_value(i), 'predicted': True, 'original': None} 
                   for i in range(31)]
    
    
    def _generate_data(self):
        """
        生成数据（与main.py的generate_data方法一致）
        
        Returns:
            np.ndarray: 生成的31维特征数据
        """
        # 使用模式3: API获取数据 + 随机值填充缺失值
        api_data = self._fetch_api_data()
        return self._fill_missing_data(api_data, fill_mode=3)
    
    def _fetch_api_data(self):
        """获取API数据（与main.py的fetch_api_data方法一致）"""
        try:
            api_data = self.api_client.get_latest_features()
            
            # 检查数据是否与上一次相同
            if self.last_api_data is not None:
                if self._is_data_same(api_data, self.last_api_data):
                    print("⚠️  检测到API数据与上一次相同，认为没有获取到新数据")
                    # 返回全None数组，让fill_missing_data处理
                    api_data = np.full(31, None)
            
            # 保存当前数据作为下次比较的基准
            self.last_api_data = api_data.copy() if api_data is not None else None
            
            return api_data
            
        except Exception as e:
            print(f"❌ API数据获取失败: {e}")
            return np.full(31, None)
    
    def _generate_mock_data(self):
        """生成模拟数据 - API不可用时100%生成模拟数据"""
        # 当API不可用时，直接生成完整的模拟数据
        # 所有31个特征都有值，模拟真实API数据
        # 每次生成略有不同的数据，避免重复检测问题
        
        data = []
        for i in range(31):
            # 添加时间戳影响，确保每次生成略有不同
            base_value = self._generate_realistic_value(i)
            # 添加小幅随机变化（±2%）
            variation = base_value * 0.02 * (random.random() - 0.5) * 2
            value = base_value + variation
            data.append(value)
        
        print("📡 模拟API返回完整数据（API不可用模式）")
        return data
    
    def _generate_realistic_value(self, feature_index):
        """生成符合特征类型的真实值"""
        if feature_index < 5:  # 贯入度和推进压力 (MPa)
            return random.uniform(0.5, 3.0)
        elif feature_index < 9:  # 土舱土压 (MPa)
            return random.uniform(0.1, 0.8)
        elif feature_index < 13:  # 推进千斤顶速度 (mm/min)
            return random.uniform(10, 50)
        elif feature_index == 13:  # 推进油缸总推力 (kN)
            return random.uniform(5000, 25000)
        elif feature_index < 18:  # 推进千斤顶行程 (mm)
            return random.uniform(100, 1500)
        elif feature_index < 21:  # 推进平均速度, 刀盘转速, 刀盘扭矩
            return random.uniform(5, 30)
        else:  # 刀盘电机扭矩 (%)
            return random.uniform(20, 80)
    
    def _process_data_with_smart_filling(self, raw_data):
        """处理数据 - 智能填充逻辑（API不可用时100%使用模拟数据）"""
        processed_data = []
        
        for i, current_value in enumerate(raw_data):
            if current_value is None:
                # 缺失数据，使用模拟数据填充
                filled_value = self._generate_filled_value(i, None)
                processed_data.append({
                    'value': filled_value,
                    'predicted': True,
                    'original': None,
                    'reason': 'missing'
                })
            else:
                # 有API数据，直接使用API数据，标记为真实数据
                processed_data.append({
                    'value': current_value,
                    'predicted': False,
                    'original': current_value,
                    'reason': 'api_data'
                })
        
        return processed_data
    
    
    
    def _generate_filled_value(self, feature_id, original_value):
        """生成填充值"""
        # 基于原始值或生成合理的默认值
        if original_value is not None:
            base_value = original_value
        else:
            base_value = self._generate_realistic_value(feature_id)
        
        # 添加小幅随机变化，模拟真实填充
        variation = base_value * 0.05 * (random.random() - 0.5) * 2  # ±5%变化
        filled_value = base_value + variation
        
        # 确保在合理范围内
        min_value = max(0, base_value * 0.7)
        max_value = base_value * 1.3
        
        return max(min_value, min(max_value, filled_value))
    
    def _get_current_api_data(self):
        """获取当前时刻的API数据 - 与main.py逻辑一致，每分钟第10秒拉取"""
        current_time = datetime.now()
        current_second = current_time.second
        
        # 检查是否到了每分钟的第10秒
        if current_second != self.DATA_FETCH_SECOND:
            # 不是拉取时间，返回None表示不更新数据
            return None
        
        # 检查是否已经在这一分钟拉取过数据（避免重复拉取）
        if (self.last_fetch_time is not None and 
            current_time.minute == self.last_fetch_time.minute and
            current_time.hour == self.last_fetch_time.hour):
            print(f"⏰ 时间: {current_time.strftime('%H:%M:%S')} - 本分钟已拉取过数据，跳过")
            return None
        
        print(f"🕐 时间: {current_time.strftime('%H:%M:%S')} - 开始拉取数据")
        self.last_fetch_time = current_time
        
        if self.api_client and self.model_predictor:
            # 使用与main.py相同的generate_data逻辑
            try:
                # 调用generate_data方法（与main.py一致）
                new_data = self._generate_data()
                data_sources = ['api'] * 31  # 标记为API数据
                return np.array(new_data), data_sources
                
            except Exception as e:
                print(f"❌ 数据生成失败: {e}")
                # 使用模拟数据填充（与main.py逻辑一致）
                api_data = self._generate_simulated_data()
                data_sources = ['simulated'] * 31
                return np.array(api_data), data_sources
        else:
            # API不可用，生成模拟数据
            print("📡 生成模拟数据（API不可用模式）")
            return self._generate_mock_data()
    
    def _is_data_same(self, data1, data2):
        """比较两次API数据是否相同 - 与main.py逻辑一致"""
        if data1 is None and data2 is None:
            return True
        if data1 is None or data2 is None:
            return False
        
        # 比较有效数据（非None值）
        for i in range(31):
            val1 = data1[i]
            val2 = data2[i]
            
            # 如果两个值都是None，认为相同
            if val1 is None and val2 is None:
                continue
            
            # 如果一个是None另一个不是，认为不同
            if val1 is None or val2 is None:
                return False
            
            # 比较数值（考虑浮点数精度）
            if abs(val1 - val2) > 1e-6:
                return False
        
        return True
    
    def _detect_data_changes(self, current_data):
        """
        检测数据变化并输出提示
        
        Args:
            current_data: 当前拉取的数据
            
        Returns:
            bool: 是否有数据变化
        """
        if self.last_fetched_data is None:
            # 第一次拉取，没有比较基准
            self.last_fetched_data = current_data.copy() if hasattr(current_data, 'copy') else current_data[:]
            return False
        
        # 比较数据变化
        changes = []
        for i in range(min(len(current_data), len(self.last_fetched_data))):
            if current_data[i] is not None and self.last_fetched_data[i] is not None:
                if abs(current_data[i] - self.last_fetched_data[i]) > 1e-6:
                    changes.append({
                        'index': i + 1,
                        'name': FEATURE_NAMES[i] if i < len(FEATURE_NAMES) else f'特征{i+1}',
                        'old_value': self.last_fetched_data[i],
                        'new_value': current_data[i],
                        'change': current_data[i] - self.last_fetched_data[i],
                        'change_percent': ((current_data[i] - self.last_fetched_data[i]) / abs(self.last_fetched_data[i])) * 100 if self.last_fetched_data[i] != 0 else 0
                    })
        
        if changes:
            print(f"🔄 检测到数据变化！共 {len(changes)} 个特征发生变化:")
            for change in changes[:5]:  # 只显示前5个变化
                direction = "↗️" if change['change'] > 0 else "↘️"
                print(f"   {direction} {change['name']}: {change['old_value']:.6f} → {change['new_value']:.6f} "
                      f"({change['change']:+.6f}, {change['change_percent']:+.1f}%)")
            
            if len(changes) > 5:
                print(f"   ... 还有 {len(changes) - 5} 个特征发生变化")
            
            # 更新存储的数据
            self.last_fetched_data = current_data.copy() if hasattr(current_data, 'copy') else current_data[:]
            return True
        else:
            print("📊 数据无变化")
            return False
    
    def _determine_tbm_status(self, current_data):
        """
        根据刀盘扭矩判定盾构机状态
        
        Args:
            current_data: 当前数据数组（31个特征）
            
        Returns:
            str: 'active' (掘进中) 或 'rest' (休息)
        """
        if current_data is None or len(current_data) < 21:
            self.tbm_status = 'rest'
            return 'rest'
        
        # 刀盘扭矩是第21个特征（索引20）
        torque_value = current_data[20]
        
        # 修改判断逻辑：刀盘扭矩大于0则为掘进中，否则为休息
        if torque_value is not None and torque_value > 0:
            if self.tbm_status != 'active':
                print(f"🔧 盾构机状态: 休息 → 掘进中 (刀盘扭矩: {torque_value:.2f} kN·m)")
            self.tbm_status = 'active'
            return 'active'
        else:
            if self.tbm_status != 'rest':
                print(f"🔧 盾构机状态: 掘进中 → 休息 (刀盘扭矩: {torque_value if torque_value is not None else 'None'})")
            self.tbm_status = 'rest'
            return 'rest'
    
    def _generate_smart_fill_data(self, current_data):
        """
        生成智能填充数据（当盾构机休息时使用）
        
        Args:
            current_data: 当前数据数组
            
        Returns:
            np.ndarray: 智能填充的31个特征值
        """
        filled_data = []
        
        for i in range(31):
            if current_data[i] is not None:
                # 有真实数据，使用真实数据
                filled_data.append(current_data[i])
            else:
                # 无真实数据，使用智能填充
                filled_data.append(self._generate_realistic_value(i))
        
        return np.array(filled_data)
    
    def _update_buffer(self, new_data):
        """更新滑动窗口缓冲区 - 与main.py逻辑一致"""
        # 左移数据
        self.buffer[:-1] = self.buffer[1:]
        
        # 添加新数据 - 处理None值
        if new_data is not None and len(new_data) == 31:
            # 将None值替换为0，避免nan问题
            processed_data = []
            for val in new_data:
                if val is None:
                    processed_data.append(0.0)
                elif isinstance(val, (int, float)):
                    # 检查是否为nan或inf
                    if np.isnan(val) if hasattr(np, 'isnan') else (val != val):
                        processed_data.append(0.0)
                    elif np.isinf(val) if hasattr(np, 'isinf') else (val == float('inf') or val == float('-inf')):
                        processed_data.append(0.0)
                    else:
                        processed_data.append(float(val))
                else:
                    processed_data.append(0.0)
            self.buffer[-1] = processed_data
        else:
            # 如果新数据为None或长度不正确，用0填充
            self.buffer[-1] = [0.0] * 31
            if new_data is not None and len(new_data) != 31:
                print(f"⚠️  新数据长度不正确: {len(new_data)}/31，使用0填充")
        
        self.step_count += 1
    
    def _predict_next_step(self):
        """预测下一时刻 - 与main.py逻辑一致"""
        if not self.model_predictor or not self.model_predictor.is_loaded:
            return None
        
        try:
            # 使用滑动窗口数据进行预测
            prediction = self.model_predictor.predict(self.buffer)
            self.last_prediction = prediction.copy()  # 保存预测结果
            return prediction
        except Exception as e:
            print(f"❌ 预测执行失败: {e}")
            return None

# 全局数据生成器
data_generator = DataGenerator()

def data_collection_thread():
    """数据收集线程"""
    global is_running, last_data
    
    while is_running:
        try:
            # 获取最新数据
            data_result = data_generator.get_latest_data()
            
            # 提取current_values
            if isinstance(data_result, dict) and 'current_values' in data_result:
                current_data = data_result['current_values']
            else:
                print(f"⚠️  数据格式错误: {type(data_result)}")
                current_data = [None] * 31
            
            # 检测数据变化
            if current_data and len(current_data) == 31:
                data_generator._detect_data_changes(current_data)
            
            # 更新全局数据
            last_data = {
                'timestamp': datetime.now().isoformat(),
                'features': current_data,
                'step_count': data_result.get('step_count', 0) if isinstance(data_result, dict) else 0,
                'buffer_ready': data_result.get('buffer_ready', False) if isinstance(data_result, dict) else False
            }
            
            # 更新last_values用于重复检测
            if current_data and len(current_data) == 31:
                data_generator.last_values = []
                for item in current_data:
                    if isinstance(item, (int, float)) and not (np.isnan(item) if hasattr(np, 'isnan') else False):
                        data_generator.last_values.append(item)
                    else:
                        data_generator.last_values.append(None)
            
            # 将数据放入缓冲区
            if not data_buffer.full():
                data_buffer.put(last_data)
            else:
                # 缓冲区满时，移除最旧的数据
                try:
                    data_buffer.get_nowait()
                    data_buffer.put(last_data)
                except queue.Empty:
                    pass
            
            print(f"✅ 数据收集完成: {datetime.now().strftime('%H:%M:%S')}")
            
        except Exception as e:
            print(f"❌ 数据收集错误: {e}")
        
        # 等待10秒（减少数据收集频率）
        time.sleep(10)

# API路由
@app.route('/')
def index():
    """提供主页面"""
    return send_from_directory('.', 'index.html')

@app.route('/api/tbm-data')
def get_tbm_data():
    """获取TBM数据API - 返回当前值和预测值"""
    global last_data
    
    try:
        print(f"📡 API请求: /api/tbm-data at {datetime.now().strftime('%H:%M:%S')}")
        
        # 获取最新数据（包含当前值和预测值）
        data_result = data_generator.get_latest_data()
        
        # 验证数据结果
        if not isinstance(data_result, dict):
            raise ValueError("数据生成器返回格式错误")
        
        # 构建返回数据
        current_values = data_result.get('current_values', [])
        current_sources = data_result.get('current_sources', [])
        prediction_values = data_result.get('prediction_values')
        step_count = data_result.get('step_count', 0)
        buffer_ready = data_result.get('buffer_ready', False)
        
        # 确保current_values是31个元素的数组
        if not isinstance(current_values, (list, np.ndarray)) or len(current_values) != 31:
            print(f"⚠️  当前值数量不正确: {len(current_values) if hasattr(current_values, '__len__') else 'N/A'}/31，使用默认值填充")
            current_values = [None] * 31
            current_sources = ['simulated'] * 31
        
        # 清理数据，处理infinity和nan值
        cleaned_values = []
        for i, val in enumerate(current_values):
            if val is None:
                cleaned_values.append(None)
            elif isinstance(val, (int, float)):
                if np.isnan(val) if hasattr(np, 'isnan') else (val != val):
                    cleaned_values.append(None)
                elif np.isinf(val) if hasattr(np, 'isinf') else (val == float('inf') or val == float('-inf')):
                    cleaned_values.append(None)
                else:
                    cleaned_values.append(val)
            else:
                cleaned_values.append(None)
        
        current_values = cleaned_values
        
        # 构建特征数据数组
        features = []
        for i in range(31):
            try:
                # 当前值 - 安全转换为Python原生类型
                current_val = current_values[i] if i < len(current_values) and current_values[i] is not None else None
                if current_val is not None:
                    current_val = float(current_val)  # 转换为Python float
                
                # 当前值数据来源
                current_source = current_sources[i] if i < len(current_sources) else 'simulated'
                
                # 预测值（如果缓冲区已满且有预测结果）
                pred_val = None
                if prediction_values is not None and i < len(prediction_values) and prediction_values[i] is not None:
                    pred_val = float(prediction_values[i])  # 转换为Python float
                
                features.append({
                    'id': i + 1,
                    'current_value': current_val,
                    'current_source': current_source,
                    'prediction_value': pred_val,
                    'step_count': int(step_count),
                    'buffer_ready': bool(buffer_ready)
                })
            except (ValueError, TypeError, IndexError) as e:
                print(f"⚠️  处理特征{i+1}时出错: {e}")
                features.append({
                    'id': i + 1,
                    'current_value': None,
                    'current_source': 'simulated',
                    'prediction_value': None,
                    'step_count': int(step_count),
                    'buffer_ready': bool(buffer_ready)
                })
        
        last_data = {
            'timestamp': datetime.now().isoformat(),
            'features': features,
            'step_count': int(step_count),
            'buffer_ready': bool(buffer_ready)
        }
        
        print(f"✅ 返回数据: 步骤{step_count}, 缓冲区{'就绪' if buffer_ready else '未就绪'}")
        return jsonify(last_data)
        
    except Exception as e:
        print(f"❌ API处理错误: {e}")
        # 返回错误响应
        error_response = {
            'error': True,
            'message': str(e),
            'timestamp': datetime.now().isoformat(),
            'features': []
        }
        return jsonify(error_response), 500

@app.route('/api/status')
def get_status():
    """获取系统状态API"""
    return jsonify({
        'status': 'running' if is_running else 'stopped',
        'api_available': API_AVAILABLE,
        'data_mode': data_generator.data_mode,
        'last_update': last_data['timestamp'] if last_data else None
    })

@app.route('/api/features')
def get_features():
    """获取特征配置API"""
    features = []
    for i, (name, unit) in enumerate(zip(FEATURE_NAMES, FEATURE_UNITS)):
        features.append({
            'id': i + 1,
            'name': name,
            'unit': unit
        })
    
    return jsonify({'features': features})

@app.route('/api/history')
def get_history():
    """获取历史数据API"""
    history = []
    
    # 从缓冲区获取历史数据
    temp_queue = queue.Queue()
    while not data_buffer.empty():
        data = data_buffer.get()
        history.append(data)
        temp_queue.put(data)
    
    # 将数据放回缓冲区
    while not temp_queue.empty():
        data_buffer.put(temp_queue.get())
    
    return jsonify({'history': history})

@app.route('/api/control', methods=['POST'])
def control_system():
    """控制系统API"""
    global is_running
    
    data = request.get_json()
    action = data.get('action')
    
    if action == 'start':
        if not is_running:
            is_running = True
            thread = threading.Thread(target=data_collection_thread, daemon=True)
            thread.start()
            return jsonify({'status': 'started'})
        else:
            return jsonify({'status': 'already_running'})
    
    elif action == 'stop':
        is_running = False
        return jsonify({'status': 'stopped'})
    
    elif action == 'set_mode':
        mode = data.get('mode', 4)
        data_generator.data_mode = mode
        return jsonify({'status': 'mode_updated', 'mode': mode})
    
    return jsonify({'error': 'Invalid action'}), 400

# 静态文件服务
@app.route('/<path:filename>')
def static_files(filename):
    """提供静态文件"""
    return send_from_directory('.', filename)

def main():
    """主函数"""
    global is_running
    
    print("🚀 启动TBM盾构机关键掘进参数实时预测系统...")
    print("=" * 50)
    
    # 启动数据收集线程
    is_running = True
    thread = threading.Thread(target=data_collection_thread, daemon=True)
    thread.start()
    
    print("✅ 数据收集线程已启动")
    print("✅ Web服务器启动中...")
    print("=" * 50)
    print("🌐 访问地址: http://localhost:5000")
    print("📊 API文档: http://localhost:5000/api/status")
    print("=" * 50)
    
    try:
        # 启动Flask应用
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
    except KeyboardInterrupt:
        print("\n🛑 正在停止系统...")
        is_running = False
        print("✅ 系统已停止")

if __name__ == '__main__':
    main()

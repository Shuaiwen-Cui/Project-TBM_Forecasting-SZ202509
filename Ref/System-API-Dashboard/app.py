#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TBM盾构机实时监控系统 - 后端API服务器

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
    
    def get_latest_data(self):
        """获取最新的TBM数据"""
        if self.api_client and self.model_predictor:
            raw_data = self._get_real_data()
        else:
            raw_data = self._generate_mock_data()
        
        # 应用智能填充逻辑（与前端一致）
        processed_data = self._process_data_with_smart_filling(raw_data)
        return processed_data
    
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
    
    def _fill_missing_data(self, api_data):
        """填充缺失数据"""
        filled_data = []
        
        for i, value in enumerate(api_data):
            if value is None:
                # 使用预测值填充
                if self.last_prediction is not None:
                    filled_data.append({
                        'value': self.last_prediction[i],
                        'predicted': True,
                        'original': None
                    })
                else:
                    # 没有预测值时使用随机值
                    filled_data.append({
                        'value': self._generate_random_value(i),
                        'predicted': True,
                        'original': None
                    })
            else:
                filled_data.append({
                    'value': value,
                    'predicted': False,
                    'original': value
                })
        
        return filled_data
    
    def _fill_with_predictions(self):
        """完全使用预测值填充"""
        if self.last_prediction is not None:
            return [{'value': pred, 'predicted': True, 'original': None} 
                   for pred in self.last_prediction]
        else:
            return [{'value': self._generate_random_value(i), 'predicted': True, 'original': None} 
                   for i in range(31)]
    
    def _generate_random_value(self, feature_index):
        """生成随机值"""
        # 根据特征类型生成合理的随机值
        if feature_index < 5:  # 贯入度和推进压力
            return random.uniform(0, 20)
        elif feature_index < 9:  # 土舱土压
            return random.uniform(0, 5)
        elif feature_index < 13:  # 推进千斤顶速度
            return random.uniform(0, 100)
        elif feature_index == 13:  # 推进油缸总推力
            return random.uniform(0, 10000)
        elif feature_index < 18:  # 推进千斤顶行程
            return random.uniform(0, 1000)
        elif feature_index < 21:  # 推进平均速度, 刀盘转速, 刀盘扭矩
            return random.uniform(0, 100)
        else:  # 刀盘电机扭矩
            return random.uniform(0, 100)
    
    def _generate_mock_data(self):
        """生成模拟数据 - 模拟真实API数据场景"""
        # 模拟API数据的特点：
        # 1. 数据通常是整批的（要么全部有，要么全部没有）
        # 2. 数据通常是几天前的历史数据
        # 3. 如果有数据，基本所有特征都有数据
        
        rand = random.random()
        
        if rand < 0.3:  # 30%概率有完整的API数据（历史数据）
            # 生成整批数据，所有特征都有值
            data = []
            for i in range(31):
                value = self._generate_realistic_value(i)
                data.append(value)  # 直接返回数值，表示真实API数据
            print("📡 模拟API返回完整历史数据")
            return data
            
        elif rand < 0.6:  # 30%概率有部分API数据（部分特征缺失）
            # 生成部分数据，模拟API部分字段缺失
            data = []
            missing_features = random.sample(range(31), random.randint(5, 15))  # 随机缺失5-15个特征
            
            for i in range(31):
                if i in missing_features:
                    data.append(None)  # 缺失数据
                else:
                    value = self._generate_realistic_value(i)
                    data.append(value)
            print(f"📡 模拟API返回部分数据，缺失{len(missing_features)}个特征")
            return data
            
        else:  # 40%概率没有API数据
            # 全部缺失，需要填充
            print("📡 模拟API无数据，需要填充")
            return [None] * 31
    
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
        """处理数据 - 智能填充逻辑（模拟真实API数据场景）"""
        processed_data = []
        
        for i, current_value in enumerate(raw_data):
            if current_value is None:
                # 缺失数据，使用预测值填充
                filled_value = self._generate_filled_value(i, None)
                processed_data.append({
                    'value': filled_value,
                    'predicted': True,
                    'original': None,
                    'reason': 'missing'
                })
            else:
                # 有API数据，但由于是历史数据，标记为填充
                # 在实际应用中，这里可能是真实的API数据
                filled_value = self._generate_filled_value(i, current_value)
                processed_data.append({
                    'value': filled_value,
                    'predicted': True,
                    'original': current_value,
                    'reason': 'historical'
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

# 全局数据生成器
data_generator = DataGenerator()

def data_collection_thread():
    """数据收集线程"""
    global is_running, last_data
    
    while is_running:
        try:
            # 获取最新数据
            current_data = data_generator.get_latest_data()
            
            # 更新全局数据
            last_data = {
                'timestamp': datetime.now().isoformat(),
                'features': current_data
            }
            
            # 更新last_values用于重复检测
            if current_data:
                data_generator.last_values = []
                for item in current_data:
                    if isinstance(item, (int, float)):
                        data_generator.last_values.append(item)
                    elif isinstance(item, dict) and 'value' in item:
                        data_generator.last_values.append(item['value'])
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
        
        # 等待2秒
        time.sleep(2)

# API路由
@app.route('/')
def index():
    """提供主页面"""
    return send_from_directory('.', 'index.html')

@app.route('/api/tbm-data')
def get_tbm_data():
    """获取TBM数据API"""
    global last_data
    
    print(f"📡 API请求: /api/tbm-data at {datetime.now().strftime('%H:%M:%S')}")
    
    if last_data is None:
        # 如果没有数据，生成模拟数据
        print("⚠️  没有缓存数据，生成模拟数据")
        raw_data = data_generator._generate_mock_data()
        processed_data = data_generator._process_data_with_smart_filling(raw_data)
        last_data = {
            'timestamp': datetime.now().isoformat(),
            'features': processed_data
        }
    
    print(f"✅ 返回数据: {len(last_data['features'])} 个特征")
    return jsonify(last_data)

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
    
    print("🚀 启动TBM盾构机实时监控系统...")
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

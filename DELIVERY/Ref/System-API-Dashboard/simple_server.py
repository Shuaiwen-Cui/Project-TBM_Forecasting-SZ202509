#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化的TBM监控服务器 - 用于测试
"""

import json
import random
import time
from datetime import datetime
from http.server import HTTPServer, SimpleHTTPRequestHandler
import threading

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
    'MPa', 'MPa', 'MPa', 'MPa', 'MPa',
    'MPa', 'MPa', 'MPa', 'MPa',
    'mm/min', 'mm/min', 'mm/min', 'mm/min',
    'kN',
    'mm', 'mm', 'mm', 'mm',
    'mm/min', 'r/min', 'kN·m',
    '%', '%', '%', '%', '%', '%', '%', '%', '%', '%'
]

class TBMRequestHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        # 添加CORS头
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()
    
    def do_GET(self):
        if self.path == '/api/tbm-data':
            self.handle_tbm_data()
        elif self.path == '/api/status':
            self.handle_status()
        elif self.path == '/api/features':
            self.handle_features()
        else:
            super().do_GET()
    
    def handle_tbm_data(self):
        """处理TBM数据请求 - 模拟真实API数据场景"""
        print(f"📡 收到数据请求: {datetime.now().strftime('%H:%M:%S')}")
        
        # 模拟API数据的特点：
        # 1. 数据通常是整批的（要么全部有，要么全部没有）
        # 2. 数据通常是几天前的历史数据
        # 3. 如果有数据，基本所有特征都有数据
        
        rand = random.random()
        
        if rand < 0.3:  # 30%概率有完整的API数据（历史数据）
            # 生成整批数据，所有特征都有值
            features = []
            for i in range(31):
                value = self._generate_realistic_value(i)
                features.append(value)  # 直接返回数值，表示真实API数据
            print("📡 模拟API返回完整历史数据")
            
        elif rand < 0.6:  # 30%概率有部分API数据（部分特征缺失）
            # 生成部分数据，模拟API部分字段缺失
            features = []
            missing_features = random.sample(range(31), random.randint(5, 15))  # 随机缺失5-15个特征
            
            for i in range(31):
                if i in missing_features:
                    features.append(None)  # 缺失数据
                else:
                    value = self._generate_realistic_value(i)
                    features.append(value)
            print(f"📡 模拟API返回部分数据，缺失{len(missing_features)}个特征")
            
        else:  # 40%概率没有API数据
            # 全部缺失，需要填充
            features = [None] * 31
            print("📡 模拟API无数据，需要填充")
        
        # 应用智能填充逻辑
        processed_features = self._process_data_with_smart_filling(features)
        
        data = {
            'timestamp': datetime.now().isoformat(),
            'features': processed_features
        }
        
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(data, ensure_ascii=False).encode('utf-8'))
        print(f"✅ 返回数据: {len(processed_features)} 个特征")
    
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
    
    def handle_status(self):
        """处理状态请求"""
        data = {
            'status': 'running',
            'api_available': True,
            'data_mode': 4,
            'last_update': datetime.now().isoformat()
        }
        
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(data, ensure_ascii=False).encode('utf-8'))
    
    def handle_features(self):
        """处理特征配置请求"""
        features = []
        for i, (name, unit) in enumerate(zip(FEATURE_NAMES, FEATURE_UNITS)):
            features.append({
                'id': i + 1,
                'name': name,
                'unit': unit
            })
        
        data = {'features': features}
        
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(data, ensure_ascii=False).encode('utf-8'))

def main():
    port = 8000
    server_address = ('', port)
    httpd = HTTPServer(server_address, TBMRequestHandler)
    
    print("🚀 启动简化TBM监控服务器...")
    print(f"🌐 访问地址: http://localhost:{port}")
    print(f"📊 测试页面: http://localhost:{port}/test.html")
    print(f"📊 主页面: http://localhost:{port}/index.html")
    print("=" * 50)
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n🛑 正在停止服务器...")
        httpd.shutdown()
        print("✅ 服务器已停止")

if __name__ == '__main__':
    main()

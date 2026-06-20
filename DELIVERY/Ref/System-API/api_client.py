#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TBM API数据获取模块 (api_client.py)
==================================

模块功能:
- 从盾构机API接口获取实时数据
- 解析API返回的JSON数据
- 提取31个特征对应的传输名数据
- 处理数据缺失和异常情况

模块职责:
- 封装API调用相关操作
- 提供统一的数据获取接口
- 处理数据解析和转换
"""

import requests
import json
import numpy as np
from datetime import datetime, timedelta
import time
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# API配置
# =============================================================================
class APIConfig:
    """API配置类"""
    BASE_URL = "https://szqhx.sinoccdc.com/qhx/shield/data/list"
    ACCESS_TOKEN = "0c46137497ffdf0369037ada468fe2d3"
    TBM_ID = "THDG24493"
    TIMEOUT = 30  # 请求超时时间(秒)
    
    # 特征名到传输名的映射（已修正，所有31个特征都有对应传输名）
    FEATURE_MAPPING = {
        '贯入度': 'date120',
        '推进区间的压力（上）': 'date16',
        '推进区间的压力（右）': 'date17',
        '推进区间的压力（下）': 'date18',
        '推进区间的压力（左）': 'date19',
        '土舱土压（右）': 'date29',
        '土舱土压（右下）': 'date30',
        '土舱土压（左）': 'date31',
        '土舱土压（左下）': 'date32',
        'No.16推进千斤顶速度': 'date7',
        'No.4推进千斤顶速度': 'date8',
        'No.8推进千斤顶速度': 'date9',
        'No.12推进千斤顶速度': 'date10',
        '推进油缸总推力': 'date12',
        'No.16推进千斤顶行程': 'date3',
        'No.4推进千斤顶行程': 'date4',
        'No.8推进千斤顶行程': 'date5',
        'No.12推进千斤顶行程': 'date6',
        '推进平均速度': 'date78',
        '刀盘转速': 'date76',
        '刀盘扭矩': 'date77',
        'No.1刀盘电机扭矩': 'date47',
        'No.2刀盘电机扭矩': 'date48',
        'No.3刀盘电机扭矩': 'date49',
        'No.4刀盘电机扭矩': 'date50',
        'No.5刀盘电机扭矩': 'date51',
        'No.6刀盘电机扭矩': 'date52',
        'No.7刀盘电机扭矩': 'date53',
        'No.8刀盘电机扭矩': 'date54',
        'No.9刀盘电机扭矩': 'date55',
        'No.10刀盘电机扭矩': 'date56'
    }

# =============================================================================
# API客户端类
# =============================================================================
class TBMAPIClient:
    """
    TBM API客户端
    
    职责:
    1. 调用盾构机数据API接口
    2. 解析API返回的JSON数据
    3. 提取31个特征对应的数据
    4. 处理API调用异常和错误
    """
    
    def __init__(self):
        """初始化API客户端"""
        self.session = requests.Session()
        self.config = APIConfig()
        self.last_request_time = 0
        self.request_interval = 1  # 请求间隔(秒)
    
    # =========================================================================
    # API调用模块
    # =========================================================================
    def fetch_latest_data(self, limit=1):
        """
        获取最新数据
        
        Args:
            limit (int): 限制返回条数
            
        Returns:
            dict: API响应数据，失败返回None
        """
        try:
            # 控制请求频率
            current_time = time.time()
            if current_time - self.last_request_time < self.request_interval:
                time.sleep(self.request_interval - (current_time - self.last_request_time))
            
            # 构建请求参数
            params = {
                "access-token": self.config.ACCESS_TOKEN
            }
            
            # 构建请求体 - 添加时间范围参数
            from datetime import datetime, timedelta
            end_time = datetime.now()
            begin_time = end_time - timedelta(days=7)  # 查询最近7天的数据
            
            payload = {
                "tbmId": self.config.TBM_ID,
                "beginTime": begin_time.strftime("%Y-%m-%d %H:%M:%S"),
                "endTime": end_time.strftime("%Y-%m-%d %H:%M:%S"),
                "limit": limit
            }
            
            # print(f"📡 请求参数: {payload}")  # 注释掉调试信息
            
            # 发送POST请求
            response = self.session.post(
                self.config.BASE_URL,
                params=params,
                json=payload,
                timeout=self.config.TIMEOUT
            )
            
            self.last_request_time = time.time()
            
            # 检查响应状态
            if response.status_code == 200:
                data = response.json()
                if data.get('code') == 200 and 'data' in data:
                    return data
                else:
                    print(f"❌ API返回错误: {data.get('msg', '未知错误')}")
                    return None
            else:
                print(f"❌ HTTP请求失败: {response.status_code}")
                return None
                
        except requests.exceptions.RequestException as e:
            print(f"❌ 请求异常: {e}")
            return None
        except json.JSONDecodeError as e:
            print(f"❌ JSON解析错误: {e}")
            return None
        except Exception as e:
            print(f"❌ 未知错误: {e}")
            return None
    
    def fetch_data_by_time_range(self, begin_time, end_time, limit=None):
        """
        按时间范围获取数据
        
        Args:
            begin_time (str): 开始时间 "2025-09-07 24:00:00"
            end_time (str): 结束时间 "2025-09-09 24:00:00"
            limit (int, optional): 限制返回条数
            
        Returns:
            dict: API响应数据，失败返回None
        """
        try:
            # 控制请求频率
            current_time = time.time()
            if current_time - self.last_request_time < self.request_interval:
                time.sleep(self.request_interval - (current_time - self.last_request_time))
            
            # 构建请求参数
            params = {
                "access-token": self.config.ACCESS_TOKEN
            }
            
            # 构建请求体
            payload = {
                "tbmId": self.config.TBM_ID,
                "beginTime": begin_time,
                "endTime": end_time
            }
            
            if limit:
                payload["limit"] = limit
            
            # 发送POST请求
            response = self.session.post(
                self.config.BASE_URL,
                params=params,
                json=payload,
                timeout=self.config.TIMEOUT
            )
            
            self.last_request_time = time.time()
            
            # 检查响应状态
            if response.status_code == 200:
                data = response.json()
                if data.get('code') == 200 and 'data' in data:
                    return data
                else:
                    print(f"❌ API返回错误: {data.get('msg', '未知错误')}")
                    return None
            else:
                print(f"❌ HTTP请求失败: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"❌ 获取数据失败: {e}")
            return None
    
    # =========================================================================
    # 数据解析模块
    # =========================================================================
    def parse_data_record(self, record):
        """
        解析单条数据记录
        
        Args:
            record (dict): API返回的单条数据记录
            
        Returns:
            dict: 解析后的数据字典
        """
        try:
            # 解析JSON数据字段
            data_json = json.loads(record.get('data', '{}'))
            
            # 创建解析结果
            parsed_data = {
                'id': record.get('id'),
                'tbm_id': record.get('tbmId'),
                'create_time': record.get('createTime'),
                'origin_time': record.get('originTime'),
                'my_time': record.get('myTime'),
                'raw_data': data_json
            }
            
            return parsed_data
            
        except Exception as e:
            print(f"❌ 解析数据记录失败: {e}")
            return None
    
    def extract_feature_values(self, parsed_data):
        """
        提取31个特征对应的数值
        
        Args:
            parsed_data (dict): 解析后的数据
            
        Returns:
            np.ndarray: 31个特征值数组，缺失值用None表示
        """
        import numpy as np
        
        if not parsed_data or 'raw_data' not in parsed_data:
            return np.full(31, None)
        
        raw_data = parsed_data['raw_data']
        feature_values = np.full(31, None)
        
        # 特征名称列表（按顺序）
        feature_names = [
            '贯入度', '推进区间的压力（上）', '推进区间的压力（右）', '推进区间的压力（下）', '推进区间的压力（左）',
            '土舱土压（右）', '土舱土压（右下）', '土舱土压（左）', '土舱土压（左下）',
            'No.16推进千斤顶速度', 'No.4推进千斤顶速度', 'No.8推进千斤顶速度', 'No.12推进千斤顶速度',
            '推进油缸总推力', 'No.16推进千斤顶行程', 'No.4推进千斤顶行程', 'No.8推进千斤顶行程', 'No.12推进千斤顶行程',
            '推进平均速度', '刀盘转速', '刀盘扭矩',
            'No.1刀盘电机扭矩', 'No.2刀盘电机扭矩', 'No.3刀盘电机扭矩', 'No.4刀盘电机扭矩', 'No.5刀盘电机扭矩',
            'No.6刀盘电机扭矩', 'No.7刀盘电机扭矩', 'No.8刀盘电机扭矩', 'No.9刀盘电机扭矩', 'No.10刀盘电机扭矩'
        ]
        
        for i, feature_name in enumerate(feature_names):
            transmission_name = self.config.FEATURE_MAPPING.get(feature_name)
            
            if transmission_name and transmission_name in raw_data:
                # 提取数值部分（去掉单位）
                value_str = raw_data[transmission_name]
                try:
                    # 提取括号前的数值
                    if '(' in value_str:
                        numeric_value = float(value_str.split('(')[0])
                    else:
                        numeric_value = float(value_str)
                    feature_values[i] = numeric_value
                except ValueError:
                    print(f"⚠️  特征{i+1} ({feature_name}) 数值解析失败: {value_str}")
                    feature_values[i] = None
            else:
                # 没有对应的传输名或数据缺失
                feature_values[i] = None
        
        return feature_values
    
    # =========================================================================
    # 数据获取接口
    # =========================================================================
    def get_latest_features(self):
        """
        获取最新的31个特征数据
        
        Returns:
            np.ndarray: 31个特征值数组，缺失值用None表示
        """
        # 获取最新数据
        api_response = self.fetch_latest_data(limit=1)
        
        if not api_response or not api_response.get('data'):
            print("❌ 未获取到API数据")
            return np.full(31, None)
        
        # 解析数据
        record = api_response['data'][0]
        parsed_data = self.parse_data_record(record)
        
        if not parsed_data:
            print("❌ 数据解析失败")
            return np.full(31, None)
        
        # 提取特征值
        feature_values = self.extract_feature_values(parsed_data)
        
        # 统计数据完整性
        valid_count = np.sum(feature_values != None)
        print(f"📊 获取到 {valid_count}/31 个有效特征值")
        
        if valid_count == 31:
            print("✅ 所有31个特征都成功获取到数据！")
        elif valid_count > 0:
            print(f"⚠️  有 {31-valid_count} 个特征数据缺失")
        
        return feature_values
    
    def test_connection(self):
        """
        测试API连接
        
        Returns:
            bool: 连接成功返回True
        """
        print("🔍 测试API连接...")
        
        api_response = self.fetch_latest_data(limit=1)
        
        if api_response and api_response.get('data'):
            print("✅ API连接成功")
            return True
        else:
            print("❌ API连接失败")
            return False

# =============================================================================
# 工具函数
# =============================================================================
def get_time_range_strings(hours=1):
    """
    获取时间范围字符串
    
    Args:
        hours (int): 时间范围（小时）
        
    Returns:
        tuple: (begin_time, end_time) 时间字符串元组
    """
    end_time = datetime.now()
    begin_time = end_time - timedelta(hours=hours)
    
    return begin_time.strftime("%Y-%m-%d %H:%M:%S"), end_time.strftime("%Y-%m-%d %H:%M:%S")

# =============================================================================
# 测试函数
# =============================================================================
def test_api_client():
    """测试API客户端"""
    print("🧪 测试TBM API客户端")
    print("=" * 50)
    
    client = TBMAPIClient()
    
    # 测试连接
    if client.test_connection():
        # 获取最新特征数据
        print("\n📊 获取最新特征数据:")
        features = client.get_latest_features()
        
        print(f"特征数据形状: {features.shape}")
        print(f"有效数据数量: {np.sum(features != None)}")
        print(f"缺失数据数量: {np.sum(features == None)}")
        
        # 显示前10个特征
        print("\n前10个特征值:")
        for i in range(min(10, len(features))):
            value = features[i]
            status = "✅" if value is not None else "❌"
            print(f"  特征{i+1:2d}: {value if value is not None else '缺失'} {status}")

if __name__ == "__main__":
    test_api_client()

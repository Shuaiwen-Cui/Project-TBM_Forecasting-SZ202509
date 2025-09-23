#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TBM API数据获取模块 - 增强鲁棒性版本 (api_client_robust.py)
========================================================

模块功能:
- 从盾构机API接口获取实时数据
- 解析API返回的JSON数据
- 提取31个特征对应的传输名数据
- 处理数据缺失和异常情况
- 增强的错误处理和重试机制

模块职责:
- 封装API调用相关操作
- 提供统一的数据获取接口
- 处理数据解析和转换
- 提供鲁棒的错误恢复机制
"""

import requests
import json
import numpy as np
from datetime import datetime, timedelta
import time
import warnings
import logging
from typing import Optional, Dict, List, Tuple, Any
warnings.filterwarnings('ignore')

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# =============================================================================
# API配置
# =============================================================================
class APIConfig:
    """API配置类 - 增强版"""
    BASE_URL = "https://szqhx.sinoccdc.com/qhx/shield/data/list"
    ACCESS_TOKEN = "0c46137497ffdf0369037ada468fe2d3"
    TBM_ID = "THDG24493"
    TIMEOUT = 30  # 请求超时时间(秒)
    MAX_RETRIES = 3  # 最大重试次数
    RETRY_DELAY = 1  # 重试延迟(秒)
    CONNECTION_TIMEOUT = 10  # 连接超时(秒)
    READ_TIMEOUT = 30  # 读取超时(秒)
    REQUEST_INTERVAL = 1  # 请求间隔(秒)
    
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
        '推进油缸总推力': 'date11',
        'No.16推进千斤顶行程': 'date12',
        'No.4推进千斤顶行程': 'date13',
        'No.8推进千斤顶行程': 'date14',
        'No.12推进千斤顶行程': 'date15',
        '推进平均速度': 'date20',
        '刀盘转速': 'date21',
        '刀盘扭矩': 'date22',
        'No.1刀盘电机扭矩': 'date23',
        'No.2刀盘电机扭矩': 'date24',
        'No.3刀盘电机扭矩': 'date25',
        'No.4刀盘电机扭矩': 'date26',
        'No.5刀盘电机扭矩': 'date27',
        'No.6刀盘电机扭矩': 'date28',
        'No.7刀盘电机扭矩': 'date29',
        'No.8刀盘电机扭矩': 'date30',
        'No.9刀盘电机扭矩': 'date31',
        'No.10刀盘电机扭矩': 'date32'
    }

# =============================================================================
# TBM API客户端 - 增强鲁棒性版本
# =============================================================================
class TBMAPIClientRobust:
    """
    TBM API客户端 - 增强鲁棒性版本
    
    职责:
    1. 调用盾构机数据API接口
    2. 解析API返回的JSON数据
    3. 提取31个特征对应的数据
    4. 处理API调用异常和错误
    5. 提供重试机制和错误恢复
    """
    
    def __init__(self):
        """初始化API客户端"""
        self.session = requests.Session()
        self.config = APIConfig()
        self.last_request_time = 0
        self.request_count = 0
        self.success_count = 0
        self.error_count = 0
        
        # 设置会话配置
        self.session.headers.update({
            'User-Agent': 'TBM-Forecasting-System/1.0',
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        })
        
        logger.info("TBM API客户端初始化完成")
    
    def _rate_limit(self):
        """控制请求频率"""
        current_time = time.time()
        if current_time - self.last_request_time < self.config.REQUEST_INTERVAL:
            sleep_time = self.config.REQUEST_INTERVAL - (current_time - self.last_request_time)
            logger.debug(f"请求频率控制，等待 {sleep_time:.2f} 秒")
            time.sleep(sleep_time)
    
    def _make_request(self, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        发送API请求 - 带重试机制
        
        Args:
            payload: 请求载荷
            
        Returns:
            API响应数据，失败返回None
        """
        for attempt in range(self.config.MAX_RETRIES):
            try:
                self._rate_limit()
                
                # 构建请求参数
                params = {"access-token": self.config.ACCESS_TOKEN}
                
                # 发送请求
                response = self.session.post(
                    self.config.BASE_URL,
                    params=params,
                    json=payload,
                    timeout=(self.config.CONNECTION_TIMEOUT, self.config.READ_TIMEOUT)
                )
                
                self.last_request_time = time.time()
                self.request_count += 1
                
                # 检查响应状态
                if response.status_code == 200:
                    try:
                        data = response.json()
                        if data.get('code') == 200 and 'data' in data:
                            self.success_count += 1
                            if attempt > 0:
                                logger.info(f"API请求成功 (重试 {attempt} 次后)")
                            return data
                        else:
                            error_msg = data.get('msg', '未知错误')
                            logger.warning(f"API返回错误: {error_msg}")
                            if attempt < self.config.MAX_RETRIES - 1:
                                logger.info(f"等待 {self.config.RETRY_DELAY} 秒后重试...")
                                time.sleep(self.config.RETRY_DELAY)
                                continue
                            return None
                    except json.JSONDecodeError as e:
                        logger.error(f"JSON解析失败: {e}")
                        if attempt < self.config.MAX_RETRIES - 1:
                            logger.info(f"等待 {self.config.RETRY_DELAY} 秒后重试...")
                            time.sleep(self.config.RETRY_DELAY)
                            continue
                        return None
                else:
                    logger.warning(f"HTTP请求失败: {response.status_code} - {response.reason}")
                    if attempt < self.config.MAX_RETRIES - 1:
                        logger.info(f"等待 {self.config.RETRY_DELAY} 秒后重试...")
                        time.sleep(self.config.RETRY_DELAY)
                        continue
                    return None
                    
            except requests.exceptions.Timeout:
                logger.warning(f"请求超时 (尝试 {attempt + 1}/{self.config.MAX_RETRIES})")
                if attempt < self.config.MAX_RETRIES - 1:
                    logger.info(f"等待 {self.config.RETRY_DELAY} 秒后重试...")
                    time.sleep(self.config.RETRY_DELAY)
                    continue
                return None
            except requests.exceptions.ConnectionError:
                logger.warning(f"连接错误 (尝试 {attempt + 1}/{self.config.MAX_RETRIES})")
                if attempt < self.config.MAX_RETRIES - 1:
                    logger.info(f"等待 {self.config.RETRY_DELAY} 秒后重试...")
                    time.sleep(self.config.RETRY_DELAY)
                    continue
                return None
            except requests.exceptions.RequestException as e:
                logger.warning(f"网络请求异常: {e} (尝试 {attempt + 1}/{self.config.MAX_RETRIES})")
                if attempt < self.config.MAX_RETRIES - 1:
                    logger.info(f"等待 {self.config.RETRY_DELAY} 秒后重试...")
                    time.sleep(self.config.RETRY_DELAY)
                    continue
                return None
            except Exception as e:
                logger.error(f"未知错误: {e} (尝试 {attempt + 1}/{self.config.MAX_RETRIES})")
                if attempt < self.config.MAX_RETRIES - 1:
                    logger.info(f"等待 {self.config.RETRY_DELAY} 秒后重试...")
                    time.sleep(self.config.RETRY_DELAY)
                    continue
                return None
        
        self.error_count += 1
        logger.error("所有重试失败，放弃请求")
        return None
    
    def fetch_latest_data(self, limit: int = 1) -> Optional[Dict[str, Any]]:
        """
        获取最新数据
        
        Args:
            limit: 限制返回条数
            
        Returns:
            API响应数据，失败返回None
        """
        try:
            # 构建请求体
            end_time = datetime.now()
            begin_time = end_time - timedelta(days=7)  # 查询最近7天的数据
            
            payload = {
                "tbmId": self.config.TBM_ID,
                "beginTime": begin_time.strftime("%Y-%m-%d %H:%M:%S"),
                "endTime": end_time.strftime("%Y-%m-%d %H:%M:%S"),
                "limit": limit
            }
            
            return self._make_request(payload)
            
        except Exception as e:
            logger.error(f"获取最新数据失败: {e}")
            return None
    
    def fetch_data_by_time_range(self, begin_time: str, end_time: str, limit: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """
        按时间范围获取数据
        
        Args:
            begin_time: 开始时间
            end_time: 结束时间
            limit: 限制返回条数
            
        Returns:
            API响应数据，失败返回None
        """
        try:
            payload = {
                "tbmId": self.config.TBM_ID,
                "beginTime": begin_time,
                "endTime": end_time
            }
            
            if limit:
                payload["limit"] = limit
            
            return self._make_request(payload)
            
        except Exception as e:
            logger.error(f"按时间范围获取数据失败: {e}")
            return None
    
    def parse_data_record(self, record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        解析单条数据记录
        
        Args:
            record: API返回的单条数据记录
            
        Returns:
            解析后的数据字典
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
            logger.error(f"数据记录解析失败: {e}")
            return None
    
    def extract_feature_values(self, parsed_data: Dict[str, Any]) -> np.ndarray:
        """
        提取31个特征值
        
        Args:
            parsed_data: 解析后的数据
            
        Returns:
            31个特征值数组
        """
        try:
            feature_values = np.full(31, None)
            raw_data = parsed_data.get('raw_data', {})
            
            for i, (feature_name, transmission_name) in enumerate(self.config.FEATURE_MAPPING.items()):
                if transmission_name in raw_data:
                    value = raw_data[transmission_name]
                    if value is not None and value != '':
                        try:
                            feature_values[i] = float(value)
                        except (ValueError, TypeError):
                            logger.warning(f"特征 {i+1} ({feature_name}) 值转换失败: {value}")
                            feature_values[i] = None
                    else:
                        feature_values[i] = None
                else:
                    feature_values[i] = None
            
            return feature_values
            
        except Exception as e:
            logger.error(f"特征值提取失败: {e}")
            return np.full(31, None)
    
    def get_latest_features(self) -> np.ndarray:
        """
        获取最新的31个特征数据 - 增强版本
        
        Returns:
            31个特征值数组，缺失值用None表示
        """
        try:
            # 获取最新数据
            api_response = self.fetch_latest_data(limit=1)
            
            if not api_response or not api_response.get('data'):
                logger.warning("未获取到API数据")
                return np.full(31, None)
            
            # 获取最新记录
            records = api_response['data']
            if not records:
                logger.warning("API返回空数据")
                return np.full(31, None)
            
            # 选择最新的记录（按ID排序，取最大的）
            latest_record = max(records, key=lambda x: x.get('id', 0))
            
            # 解析数据
            parsed_data = self.parse_data_record(latest_record)
            
            if not parsed_data:
                logger.error("数据解析失败")
                return np.full(31, None)
            
            # 提取特征值
            feature_values = self.extract_feature_values(parsed_data)
            
            # 统计数据完整性
            valid_count = np.sum(feature_values != None)
            logger.info(f"获取到 {valid_count}/31 个有效特征值 (记录ID: {latest_record.get('id', 'N/A')})")
            
            if valid_count == 31:
                logger.info("所有31个特征都成功获取到数据！")
            elif valid_count > 0:
                logger.warning(f"有 {31-valid_count} 个特征数据缺失")
            
            return feature_values
            
        except Exception as e:
            logger.error(f"获取最新特征数据失败: {e}")
            return np.full(31, None)
    
    def get_historical_data(self, minutes_back: int = 5) -> List[Dict[str, Any]]:
        """
        获取历史数据用于初始化缓冲区
        
        Args:
            minutes_back: 获取多少分钟前的数据
            
        Returns:
            包含历史数据记录的列表
        """
        try:
            # 计算时间范围
            end_time = datetime.now()
            start_time = end_time - timedelta(minutes=minutes_back)
            
            # 格式化时间字符串
            start_time_str = start_time.strftime("%Y-%m-%d %H:%M:%S")
            end_time_str = end_time.strftime("%Y-%m-%d %H:%M:%S")
            
            logger.info(f"获取历史数据: {start_time_str} 到 {end_time_str}")
            
            # 获取历史数据
            api_response = self.fetch_data_by_time_range(
                begin_time=start_time_str,
                end_time=end_time_str,
                limit=minutes_back
            )
            
            if not api_response or not api_response.get('data'):
                logger.warning("未获取到历史数据")
                return []
            
            records = api_response['data']
            logger.info(f"获取到 {len(records)} 条历史记录")
            
            # 解析历史数据
            historical_data = []
            for i, record in enumerate(records):
                parsed_data = self.parse_data_record(record)
                if parsed_data:
                    feature_values = self.extract_feature_values(parsed_data)
                    valid_count = np.sum(feature_values != None)
                    
                    historical_data.append({
                        'features': feature_values,
                        'timestamp': parsed_data.get('create_time'),
                        'record_id': parsed_data.get('id'),
                        'valid_count': valid_count
                    })
                    
                    logger.debug(f"记录 {i+1}: ID={parsed_data.get('id')}, 有效特征={valid_count}/31")
            
            # 按时间排序，确保顺序正确
            historical_data.sort(key=lambda x: x['record_id'])
            
            logger.info(f"成功解析 {len(historical_data)} 条历史记录")
            return historical_data
            
        except Exception as e:
            logger.error(f"获取历史数据失败: {e}")
            return []
    
    def test_connection(self) -> bool:
        """
        测试API连接
        
        Returns:
            连接成功返回True
        """
        try:
            logger.info("测试API连接...")
            
            api_response = self.fetch_latest_data(limit=1)
            
            if api_response and api_response.get('data'):
                logger.info("API连接测试成功")
                return True
            else:
                logger.warning("API连接测试失败")
                return False
                
        except Exception as e:
            logger.error(f"API连接测试异常: {e}")
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取API客户端统计信息
        
        Returns:
            统计信息字典
        """
        success_rate = (self.success_count / self.request_count * 100) if self.request_count > 0 else 0
        
        return {
            'total_requests': self.request_count,
            'successful_requests': self.success_count,
            'failed_requests': self.error_count,
            'success_rate': f"{success_rate:.2f}%",
            'last_request_time': self.last_request_time
        }
    
    def reset_statistics(self):
        """重置统计信息"""
        self.request_count = 0
        self.success_count = 0
        self.error_count = 0
        logger.info("统计信息已重置")
    
    def close(self):
        """关闭客户端，清理资源"""
        try:
            self.session.close()
            logger.info("API客户端已关闭")
        except Exception as e:
            logger.error(f"关闭API客户端时出错: {e}")

# =============================================================================
# 程序入口
# =============================================================================
if __name__ == "__main__":
    # 测试API客户端
    client = TBMAPIClientRobust()
    
    try:
        # 测试连接
        if client.test_connection():
            print("✅ API连接测试成功")
            
            # 获取最新数据
            features = client.get_latest_features()
            print(f"📊 获取到 {np.sum(features != None)}/31 个有效特征")
            
            # 获取历史数据
            historical = client.get_historical_data(minutes_back=5)
            print(f"📅 获取到 {len(historical)} 条历史记录")
            
            # 显示统计信息
            stats = client.get_statistics()
            print(f"📈 统计信息: {stats}")
        else:
            print("❌ API连接测试失败")
    
    finally:
        client.close()

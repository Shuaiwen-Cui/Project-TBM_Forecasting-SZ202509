#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据验证和异常处理模块 (data_validator.py)
========================================

模块功能:
- 数据格式验证
- 数据范围检查
- 异常数据检测和处理
- 数据质量评估

模块职责:
- 确保数据完整性和一致性
- 提供数据清洗和修复功能
- 监控数据质量指标
"""

import numpy as np
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

# 配置日志
logger = logging.getLogger(__name__)

class DataValidator:
    """数据验证器"""
    
    def __init__(self):
        """初始化数据验证器"""
        # 特征名称定义
        self.feature_names = [
            '贯入度', '推进区间的压力（上）', '推进区间的压力（右）', '推进区间的压力（下）', '推进区间的压力（左）',
            '土舱土压（右）', '土舱土压（右下）', '土舱土压（左）', '土舱土压（左下）',
            'No.16推进千斤顶速度', 'No.4推进千斤顶速度', 'No.8推进千斤顶速度', 'No.12推进千斤顶速度',
            '推进油缸总推力', 'No.16推进千斤顶行程', 'No.4推进千斤顶行程', 'No.8推进千斤顶行程', 'No.12推进千斤顶行程',
            '推进平均速度', '刀盘转速', '刀盘扭矩',
            'No.1刀盘电机扭矩', 'No.2刀盘电机扭矩', 'No.3刀盘电机扭矩', 'No.4刀盘电机扭矩', 'No.5刀盘电机扭矩',
            'No.6刀盘电机扭矩', 'No.7刀盘电机扭矩', 'No.8刀盘电机扭矩', 'No.9刀盘电机扭矩', 'No.10刀盘电机扭矩'
        ]
        
        # 特征单位定义
        self.feature_units = [
            'MPa', 'MPa', 'MPa', 'MPa', 'MPa',  # 贯入度, 推进压力
            'MPa', 'MPa', 'MPa', 'MPa',  # 土舱土压
            'mm/min', 'mm/min', 'mm/min', 'mm/min',  # 推进千斤顶速度
            'kN',  # 推进油缸总推力
            'mm', 'mm', 'mm', 'mm',  # 推进千斤顶行程
            'mm/min', 'r/min', 'kN·m',  # 推进平均速度, 刀盘转速, 刀盘扭矩
            '%', '%', '%', '%', '%', '%', '%', '%', '%', '%'  # 刀盘电机扭矩
        ]
        
        # 特征合理范围定义
        self.feature_ranges = {
            # 贯入度和推进压力 (MPa)
            0: (0.1, 5.0),  # 贯入度
            1: (0.1, 5.0),  # 推进区间的压力（上）
            2: (0.1, 5.0),  # 推进区间的压力（右）
            3: (0.1, 5.0),  # 推进区间的压力（下）
            4: (0.1, 5.0),  # 推进区间的压力（左）
            
            # 土舱土压 (MPa)
            5: (0.01, 2.0),  # 土舱土压（右）
            6: (0.01, 2.0),  # 土舱土压（右下）
            7: (0.01, 2.0),  # 土舱土压（左）
            8: (0.01, 2.0),  # 土舱土压（左下）
            
            # 推进千斤顶速度 (mm/min)
            9: (0, 100),  # No.16推进千斤顶速度
            10: (0, 100),  # No.4推进千斤顶速度
            11: (0, 100),  # No.8推进千斤顶速度
            12: (0, 100),  # No.12推进千斤顶速度
            
            # 推进油缸总推力 (kN)
            13: (1000, 20000),  # 推进油缸总推力
            
            # 推进千斤顶行程 (mm)
            14: (0, 3000),  # No.16推进千斤顶行程
            15: (0, 3000),  # No.4推进千斤顶行程
            16: (0, 3000),  # No.8推进千斤顶行程
            17: (0, 3000),  # No.12推进千斤顶行程
            
            # 推进平均速度 (mm/min)
            18: (0, 100),  # 推进平均速度
            
            # 刀盘转速 (r/min)
            19: (0, 5),  # 刀盘转速
            
            # 刀盘扭矩 (kN·m)
            20: (0, 10000),  # 刀盘扭矩
            
            # 刀盘电机扭矩 (%)
            21: (0, 100),  # No.1刀盘电机扭矩
            22: (0, 100),  # No.2刀盘电机扭矩
            23: (0, 100),  # No.3刀盘电机扭矩
            24: (0, 100),  # No.4刀盘电机扭矩
            25: (0, 100),  # No.5刀盘电机扭矩
            26: (0, 100),  # No.6刀盘电机扭矩
            27: (0, 100),  # No.7刀盘电机扭矩
            28: (0, 100),  # No.8刀盘电机扭矩
            29: (0, 100),  # No.9刀盘电机扭矩
            30: (0, 100),  # No.10刀盘电机扭矩
        }
        
        # 数据质量统计
        self.validation_stats = {
            'total_validations': 0,
            'valid_data_count': 0,
            'invalid_data_count': 0,
            'out_of_range_count': 0,
            'missing_data_count': 0,
            'anomaly_count': 0
        }
    
    def validate_feature_data(self, data: np.ndarray) -> Dict[str, Any]:
        """
        验证特征数据
        
        Args:
            data: 31个特征值数组
            
        Returns:
            验证结果字典
        """
        self.validation_stats['total_validations'] += 1
        
        if not isinstance(data, np.ndarray) or len(data) != 31:
            logger.error(f"数据格式错误: 期望31个特征，实际{len(data) if hasattr(data, '__len__') else 'N/A'}个")
            self.validation_stats['invalid_data_count'] += 1
            return {
                'is_valid': False,
                'error_type': 'format_error',
                'message': '数据格式错误',
                'valid_count': 0,
                'invalid_features': list(range(31))
            }
        
        valid_count = 0
        invalid_features = []
        out_of_range_features = []
        anomaly_features = []
        
        for i, value in enumerate(data):
            if value is None or np.isnan(value):
                invalid_features.append(i)
                self.validation_stats['missing_data_count'] += 1
                continue
            
            # 检查数值范围
            if i in self.feature_ranges:
                min_val, max_val = self.feature_ranges[i]
                if not (min_val <= value <= max_val):
                    out_of_range_features.append(i)
                    self.validation_stats['out_of_range_count'] += 1
                    logger.warning(f"特征 {i+1} ({self.feature_names[i]}) 超出范围: {value} (范围: {min_val}-{max_val})")
                    continue
            
            # 检查异常值（使用3σ原则）
            if self._is_anomaly(value, i):
                anomaly_features.append(i)
                self.validation_stats['anomaly_count'] += 1
                logger.warning(f"特征 {i+1} ({self.feature_names[i]}) 检测到异常值: {value}")
                continue
            
            valid_count += 1
        
        is_valid = valid_count >= 25  # 至少25个特征有效才认为数据可用
        
        if is_valid:
            self.validation_stats['valid_data_count'] += 1
        else:
            self.validation_stats['invalid_data_count'] += 1
        
        return {
            'is_valid': is_valid,
            'valid_count': valid_count,
            'invalid_features': invalid_features,
            'out_of_range_features': out_of_range_features,
            'anomaly_features': anomaly_features,
            'quality_score': valid_count / 31 * 100
        }
    
    def _is_anomaly(self, value: float, feature_index: int) -> bool:
        """
        检测异常值（简化版本）
        
        Args:
            value: 特征值
            feature_index: 特征索引
            
        Returns:
            是否为异常值
        """
        # 这里可以实现更复杂的异常检测算法
        # 目前使用简单的范围检查
        if feature_index in self.feature_ranges:
            min_val, max_val = self.feature_ranges[feature_index]
            # 允许超出范围20%的值
            extended_min = min_val * 0.8
            extended_max = max_val * 1.2
            return not (extended_min <= value <= extended_max)
        
        return False
    
    def clean_data(self, data: np.ndarray, validation_result: Dict[str, Any]) -> np.ndarray:
        """
        清洗数据
        
        Args:
            data: 原始数据
            validation_result: 验证结果
            
        Returns:
            清洗后的数据
        """
        cleaned_data = data.copy()
        
        # 处理缺失值
        for i in validation_result.get('invalid_features', []):
            if i in self.feature_ranges:
                min_val, max_val = self.feature_ranges[i]
                # 使用范围中值填充
                cleaned_data[i] = (min_val + max_val) / 2
                logger.info(f"特征 {i+1} 缺失值已用中值填充: {cleaned_data[i]}")
            else:
                cleaned_data[i] = 0.0
        
        # 处理超出范围的值
        for i in validation_result.get('out_of_range_features', []):
            if i in self.feature_ranges:
                min_val, max_val = self.feature_ranges[i]
                # 限制在合理范围内
                cleaned_data[i] = np.clip(cleaned_data[i], min_val, max_val)
                logger.info(f"特征 {i+1} 超出范围值已限制: {cleaned_data[i]}")
        
        # 处理异常值
        for i in validation_result.get('anomaly_features', []):
            if i in self.feature_ranges:
                min_val, max_val = self.feature_ranges[i]
                # 使用范围中值替换异常值
                cleaned_data[i] = (min_val + max_val) / 2
                logger.info(f"特征 {i+1} 异常值已用中值替换: {cleaned_data[i]}")
        
        return cleaned_data
    
    def get_validation_report(self) -> Dict[str, Any]:
        """
        获取验证报告
        
        Returns:
            验证报告字典
        """
        total = self.validation_stats['total_validations']
        if total == 0:
            return {
                'total_validations': 0,
                'valid_rate': 0,
                'quality_metrics': {}
            }
        
        valid_rate = self.validation_stats['valid_data_count'] / total * 100
        missing_rate = self.validation_stats['missing_data_count'] / (total * 31) * 100
        out_of_range_rate = self.validation_stats['out_of_range_count'] / (total * 31) * 100
        anomaly_rate = self.validation_stats['anomaly_count'] / (total * 31) * 100
        
        return {
            'total_validations': total,
            'valid_rate': f"{valid_rate:.2f}%",
            'quality_metrics': {
                'missing_data_rate': f"{missing_rate:.2f}%",
                'out_of_range_rate': f"{out_of_range_rate:.2f}%",
                'anomaly_rate': f"{anomaly_rate:.2f}%"
            },
            'raw_stats': self.validation_stats
        }
    
    def reset_statistics(self):
        """重置统计信息"""
        self.validation_stats = {
            'total_validations': 0,
            'valid_data_count': 0,
            'invalid_data_count': 0,
            'out_of_range_count': 0,
            'missing_data_count': 0,
            'anomaly_count': 0
        }
        logger.info("数据验证统计信息已重置")

# =============================================================================
# 程序入口
# =============================================================================
if __name__ == "__main__":
    # 测试数据验证器
    validator = DataValidator()
    
    # 创建测试数据
    test_data = np.array([
        1.5, 2.0, 1.8, 2.1, 1.9,  # 推进压力
        0.3, 0.4, 0.35, 0.38,  # 土舱土压
        25, 30, 28, 32,  # 推进千斤顶速度
        8000,  # 推进油缸总推力
        1500, 1600, 1550, 1580,  # 推进千斤顶行程
        30, 1.5, 3000,  # 推进平均速度, 刀盘转速, 刀盘扭矩
        50, 55, 48, 52, 58, 45, 60, 47, 53, 49  # 刀盘电机扭矩
    ])
    
    # 添加一些异常数据
    test_data[0] = 10.0  # 超出范围
    test_data[5] = None  # 缺失值
    test_data[20] = 50000  # 异常值
    
    print("🧪 测试数据验证器...")
    print(f"测试数据: {test_data}")
    
    # 验证数据
    result = validator.validate_feature_data(test_data)
    print(f"验证结果: {result}")
    
    # 清洗数据
    cleaned_data = validator.clean_data(test_data, result)
    print(f"清洗后数据: {cleaned_data}")
    
    # 获取验证报告
    report = validator.get_validation_report()
    print(f"验证报告: {report}")

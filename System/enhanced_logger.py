#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强日志记录模块 (enhanced_logger.py)
==================================

模块功能:
- 结构化日志记录
- 日志轮转和压缩
- 不同级别的日志处理
- 性能监控日志

模块职责:
- 提供统一的日志接口
- 管理日志文件
- 支持日志查询和分析
"""

import logging
import logging.handlers
import os
import json
import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from pathlib import Path

class EnhancedLogger:
    """增强日志记录器"""
    
    def __init__(self, log_dir: str = "logs", max_file_size: int = 10*1024*1024, 
                 backup_count: int = 5, log_level: str = "INFO"):
        """
        初始化增强日志记录器
        
        Args:
            log_dir: 日志目录
            max_file_size: 最大文件大小(字节)
            backup_count: 备份文件数量
            log_level: 日志级别
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # 设置日志级别
        level_map = {
            'DEBUG': logging.DEBUG,
            'INFO': logging.INFO,
            'WARNING': logging.WARNING,
            'ERROR': logging.ERROR,
            'CRITICAL': logging.CRITICAL
        }
        self.log_level = level_map.get(log_level.upper(), logging.INFO)
        
        # 创建不同的日志记录器
        self.loggers = {}
        self._setup_loggers(max_file_size, backup_count)
        
        # 性能统计
        self.performance_stats = {
            'total_logs': 0,
            'error_count': 0,
            'warning_count': 0,
            'start_time': datetime.now()
        }
    
    def _setup_loggers(self, max_file_size: int, backup_count: int):
        """设置日志记录器"""
        # 主日志记录器
        self.loggers['main'] = self._create_logger(
            'main',
            'tbm_system.log',
            max_file_size,
            backup_count
        )
        
        # 错误日志记录器
        self.loggers['error'] = self._create_logger(
            'error',
            'tbm_errors.log',
            max_file_size,
            backup_count,
            level=logging.ERROR
        )
        
        # API日志记录器
        self.loggers['api'] = self._create_logger(
            'api',
            'tbm_api.log',
            max_file_size,
            backup_count
        )
        
        # 预测日志记录器
        self.loggers['prediction'] = self._create_logger(
            'prediction',
            'tbm_prediction.log',
            max_file_size,
            backup_count
        )
        
        # 性能日志记录器
        self.loggers['performance'] = self._create_logger(
            'performance',
            'tbm_performance.log',
            max_file_size,
            backup_count
        )
    
    def _create_logger(self, name: str, filename: str, max_file_size: int, 
                      backup_count: int, level: int = None) -> logging.Logger:
        """创建日志记录器"""
        logger = logging.getLogger(name)
        logger.setLevel(level or self.log_level)
        
        # 清除现有处理器
        logger.handlers.clear()
        
        # 文件处理器（带轮转）
        file_handler = logging.handlers.RotatingFileHandler(
            self.log_dir / filename,
            maxBytes=max_file_size,
            backupCount=backup_count,
            encoding='utf-8'
        )
        
        # 控制台处理器
        console_handler = logging.StreamHandler()
        
        # 设置格式
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # 添加处理器
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        # 防止重复日志
        logger.propagate = False
        
        return logger
    
    def log_system_event(self, level: str, message: str, **kwargs):
        """记录系统事件"""
        logger = self.loggers['main']
        self._log_with_context(logger, level, message, **kwargs)
    
    def log_api_event(self, level: str, message: str, **kwargs):
        """记录API事件"""
        logger = self.loggers['api']
        self._log_with_context(logger, level, message, **kwargs)
    
    def log_prediction_event(self, level: str, message: str, **kwargs):
        """记录预测事件"""
        logger = self.loggers['prediction']
        self._log_with_context(logger, level, message, **kwargs)
    
    def log_performance_event(self, level: str, message: str, **kwargs):
        """记录性能事件"""
        logger = self.loggers['performance']
        self._log_with_context(logger, level, message, **kwargs)
    
    def log_error(self, message: str, exception: Exception = None, **kwargs):
        """记录错误"""
        logger = self.loggers['error']
        if exception:
            logger.error(f"{message}: {str(exception)}", exc_info=True, extra=kwargs)
        else:
            logger.error(message, extra=kwargs)
        
        self.performance_stats['error_count'] += 1
        self.performance_stats['total_logs'] += 1
    
    def log_warning(self, message: str, **kwargs):
        """记录警告"""
        logger = self.loggers['main']
        logger.warning(message, extra=kwargs)
        
        self.performance_stats['warning_count'] += 1
        self.performance_stats['total_logs'] += 1
    
    def _log_with_context(self, logger: logging.Logger, level: str, message: str, **kwargs):
        """带上下文的日志记录"""
        # 添加上下文信息
        context = {
            'timestamp': datetime.now().isoformat(),
            'pid': os.getpid(),
            **kwargs
        }
        
        # 记录日志
        log_method = getattr(logger, level.lower(), logger.info)
        log_method(message, extra=context)
        
        # 更新统计
        self.performance_stats['total_logs'] += 1
        
        # 如果是错误或警告，也记录到错误日志
        if level.upper() in ['ERROR', 'CRITICAL']:
            self.loggers['error'].error(message, extra=context)
            self.performance_stats['error_count'] += 1
        elif level.upper() == 'WARNING':
            self.performance_stats['warning_count'] += 1
    
    def log_structured_data(self, data_type: str, data: Dict[str, Any], level: str = "INFO"):
        """记录结构化数据"""
        message = f"[{data_type}] {json.dumps(data, ensure_ascii=False, default=str)}"
        self.log_system_event(level, message, data_type=data_type, **data)
    
    def log_api_request(self, method: str, url: str, status_code: int, 
                       response_time: float, **kwargs):
        """记录API请求"""
        data = {
            'method': method,
            'url': url,
            'status_code': status_code,
            'response_time': response_time,
            **kwargs
        }
        self.log_api_event("INFO", f"API请求: {method} {url}", **data)
    
    def log_prediction_result(self, input_data: Dict[str, Any], 
                            prediction: Dict[str, Any], **kwargs):
        """记录预测结果"""
        data = {
            'input_features': len(input_data.get('features', [])),
            'prediction_count': len(prediction.get('values', [])),
            'buffer_ready': input_data.get('buffer_ready', False),
            **kwargs
        }
        self.log_prediction_event("INFO", "预测执行完成", **data)
    
    def log_performance_metric(self, metric_name: str, value: float, 
                             unit: str = "", **kwargs):
        """记录性能指标"""
        data = {
            'metric_name': metric_name,
            'value': value,
            'unit': unit,
            **kwargs
        }
        self.log_performance_event("INFO", f"性能指标: {metric_name}={value}{unit}", **data)
    
    def get_log_statistics(self) -> Dict[str, Any]:
        """获取日志统计信息"""
        uptime = datetime.now() - self.performance_stats['start_time']
        
        return {
            'total_logs': self.performance_stats['total_logs'],
            'error_count': self.performance_stats['error_count'],
            'warning_count': self.performance_stats['warning_count'],
            'uptime_seconds': uptime.total_seconds(),
            'logs_per_minute': self.performance_stats['total_logs'] / (uptime.total_seconds() / 60),
            'error_rate': self.performance_stats['error_count'] / max(self.performance_stats['total_logs'], 1) * 100
        }
    
    def cleanup_old_logs(self, days_to_keep: int = 30):
        """清理旧日志文件"""
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        cleaned_files = 0
        
        for log_file in self.log_dir.glob("*.log*"):
            if log_file.stat().st_mtime < cutoff_date.timestamp():
                try:
                    log_file.unlink()
                    cleaned_files += 1
                except Exception as e:
                    self.log_error(f"清理日志文件失败: {log_file}", e)
        
        self.log_system_event("INFO", f"清理了 {cleaned_files} 个旧日志文件")
        return cleaned_files
    
    def get_recent_logs(self, log_type: str = "main", lines: int = 100) -> List[str]:
        """获取最近的日志"""
        log_file = self.log_dir / f"tbm_{log_type}.log"
        
        if not log_file.exists():
            return []
        
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                all_lines = f.readlines()
                return all_lines[-lines:]
        except Exception as e:
            self.log_error(f"读取日志文件失败: {log_file}", e)
            return []

# 全局日志记录器实例
enhanced_logger = EnhancedLogger()

# 便捷函数
def log_system(level: str, message: str, **kwargs):
    """记录系统日志"""
    enhanced_logger.log_system_event(level, message, **kwargs)

def log_api(level: str, message: str, **kwargs):
    """记录API日志"""
    enhanced_logger.log_api_event(level, message, **kwargs)

def log_prediction(level: str, message: str, **kwargs):
    """记录预测日志"""
    enhanced_logger.log_prediction_event(level, message, **kwargs)

def log_performance(level: str, message: str, **kwargs):
    """记录性能日志"""
    enhanced_logger.log_performance_event(level, message, **kwargs)

def log_error(message: str, exception: Exception = None, **kwargs):
    """记录错误日志"""
    enhanced_logger.log_error(message, exception, **kwargs)

# =============================================================================
# 程序入口
# =============================================================================
if __name__ == "__main__":
    # 测试增强日志记录器
    logger = EnhancedLogger()
    
    # 测试不同类型的日志
    logger.log_system_event("INFO", "系统启动")
    logger.log_api_event("INFO", "API连接成功")
    logger.log_prediction_event("INFO", "预测完成")
    logger.log_performance_event("INFO", "性能正常")
    
    # 测试结构化数据
    logger.log_structured_data("user_action", {
        "action": "login",
        "user_id": "12345",
        "timestamp": datetime.now().isoformat()
    })
    
    # 测试错误日志
    try:
        raise ValueError("测试错误")
    except Exception as e:
        logger.log_error("测试错误处理", e)
    
    # 获取统计信息
    stats = logger.get_log_statistics()
    print(f"日志统计: {stats}")

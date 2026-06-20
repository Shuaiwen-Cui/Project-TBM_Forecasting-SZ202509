#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
系统健康检查和自动恢复模块 (system_monitor.py)
==========================================

模块功能:
- 系统健康状态监控
- 自动错误检测和恢复
- 性能指标监控
- 资源使用监控

模块职责:
- 监控系统各组件状态
- 提供自动恢复机制
- 记录系统运行指标
- 预警和告警功能
"""

import time
import psutil
import logging
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum

# 配置日志
logger = logging.getLogger(__name__)

class HealthStatus(Enum):
    """健康状态枚举"""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"

@dataclass
class HealthMetric:
    """健康指标数据类"""
    name: str
    value: float
    unit: str
    status: HealthStatus
    threshold_warning: float
    threshold_critical: float
    timestamp: datetime

@dataclass
class SystemAlert:
    """系统告警数据类"""
    level: str
    component: str
    message: str
    timestamp: datetime
    resolved: bool = False

class SystemMonitor:
    """系统监控器"""
    
    def __init__(self, check_interval: int = 30):
        """
        初始化系统监控器
        
        Args:
            check_interval: 检查间隔(秒)
        """
        self.check_interval = check_interval
        self.is_running = False
        self.monitor_thread = None
        
        # 健康指标
        self.health_metrics: Dict[str, HealthMetric] = {}
        self.alerts: List[SystemAlert] = []
        
        # 性能历史记录
        self.performance_history: List[Dict[str, Any]] = []
        self.max_history_size = 1000
        
        # 恢复策略
        self.recovery_strategies: Dict[str, Callable] = {}
        
        # 监控配置
        self.monitor_config = {
            'cpu_threshold_warning': 80.0,
            'cpu_threshold_critical': 95.0,
            'memory_threshold_warning': 80.0,
            'memory_threshold_critical': 95.0,
            'disk_threshold_warning': 85.0,
            'disk_threshold_critical': 95.0,
            'api_timeout_threshold': 30.0,
            'api_error_rate_threshold': 20.0
        }
        
        logger.info("系统监控器初始化完成")
    
    def start_monitoring(self):
        """开始监控"""
        if self.is_running:
            logger.warning("监控已在运行中")
            return
        
        self.is_running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("系统监控已启动")
    
    def stop_monitoring(self):
        """停止监控"""
        self.is_running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("系统监控已停止")
    
    def _monitor_loop(self):
        """监控循环"""
        while self.is_running:
            try:
                self._check_system_health()
                self._check_api_health()
                self._check_memory_usage()
                self._check_disk_usage()
                self._cleanup_old_data()
                
                time.sleep(self.check_interval)
                
            except Exception as e:
                logger.error(f"监控循环异常: {e}")
                time.sleep(self.check_interval)
    
    def _check_system_health(self):
        """检查系统健康状态"""
        try:
            # CPU使用率
            cpu_percent = psutil.cpu_percent(interval=1)
            self._update_metric(
                'cpu_usage',
                cpu_percent,
                '%',
                self.monitor_config['cpu_threshold_warning'],
                self.monitor_config['cpu_threshold_critical']
            )
            
            # 内存使用率
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            self._update_metric(
                'memory_usage',
                memory_percent,
                '%',
                self.monitor_config['memory_threshold_warning'],
                self.monitor_config['memory_threshold_critical']
            )
            
            # 磁盘使用率
            try:
                disk = psutil.disk_usage('/')
            except OSError:
                # Windows系统使用当前目录
                disk = psutil.disk_usage('.')
            disk_percent = (disk.used / disk.total) * 100
            self._update_metric(
                'disk_usage',
                disk_percent,
                '%',
                self.monitor_config['disk_threshold_warning'],
                self.monitor_config['disk_threshold_critical']
            )
            
        except Exception as e:
            logger.error(f"系统健康检查失败: {e}")
    
    def _check_api_health(self):
        """检查API健康状态"""
        # 这里可以添加API健康检查逻辑
        # 例如检查API响应时间、错误率等
        pass
    
    def _check_memory_usage(self):
        """检查内存使用情况"""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / 1024 / 1024
            
            self._update_metric(
                'process_memory',
                memory_mb,
                'MB',
                500.0,  # 500MB警告
                1000.0  # 1GB严重
            )
            
        except Exception as e:
            logger.error(f"内存使用检查失败: {e}")
    
    def _check_disk_usage(self):
        """检查磁盘使用情况"""
        try:
            # 检查当前目录磁盘使用
            disk = psutil.disk_usage('.')
            disk_gb = disk.used / 1024 / 1024 / 1024
            
            self._update_metric(
                'disk_used',
                disk_gb,
                'GB',
                10.0,  # 10GB警告
                20.0   # 20GB严重
            )
            
        except Exception as e:
            logger.error(f"磁盘使用检查失败: {e}")
    
    def _update_metric(self, name: str, value: float, unit: str, 
                      threshold_warning: float, threshold_critical: float):
        """更新健康指标"""
        # 确定状态
        if value >= threshold_critical:
            status = HealthStatus.CRITICAL
        elif value >= threshold_warning:
            status = HealthStatus.WARNING
        else:
            status = HealthStatus.HEALTHY
        
        # 创建或更新指标
        metric = HealthMetric(
            name=name,
            value=value,
            unit=unit,
            status=status,
            threshold_warning=threshold_warning,
            threshold_critical=threshold_critical,
            timestamp=datetime.now()
        )
        
        self.health_metrics[name] = metric
        
        # 检查是否需要告警
        if status in [HealthStatus.WARNING, HealthStatus.CRITICAL]:
            self._create_alert(name, status, value, threshold_warning, threshold_critical)
    
    def _create_alert(self, component: str, status: HealthStatus, 
                     value: float, threshold_warning: float, threshold_critical: float):
        """创建告警"""
        level = "WARNING" if status == HealthStatus.WARNING else "CRITICAL"
        threshold = threshold_critical if status == HealthStatus.CRITICAL else threshold_warning
        
        alert = SystemAlert(
            level=level,
            component=component,
            message=f"{component} 当前值: {value:.2f}, 阈值: {threshold:.2f}",
            timestamp=datetime.now()
        )
        
        self.alerts.append(alert)
        logger.warning(f"系统告警: {alert.message}")
        
        # 尝试自动恢复
        self._attempt_recovery(component, status)
    
    def _attempt_recovery(self, component: str, status: HealthStatus):
        """尝试自动恢复"""
        if component in self.recovery_strategies:
            try:
                recovery_func = self.recovery_strategies[component]
                recovery_func(status)
                logger.info(f"组件 {component} 自动恢复尝试完成")
            except Exception as e:
                logger.error(f"组件 {component} 自动恢复失败: {e}")
    
    def register_recovery_strategy(self, component: str, recovery_func: Callable):
        """注册恢复策略"""
        self.recovery_strategies[component] = recovery_func
        logger.info(f"为组件 {component} 注册恢复策略")
    
    def _cleanup_old_data(self):
        """清理旧数据"""
        # 清理旧的性能历史记录
        if len(self.performance_history) > self.max_history_size:
            self.performance_history = self.performance_history[-self.max_history_size:]
        
        # 清理已解决的旧告警
        current_time = datetime.now()
        self.alerts = [
            alert for alert in self.alerts 
            if not alert.resolved or (current_time - alert.timestamp).seconds < 3600
        ]
    
    def get_health_status(self) -> Dict[str, Any]:
        """获取系统健康状态"""
        # 计算整体健康状态
        if not self.health_metrics:
            overall_status = HealthStatus.UNKNOWN
        else:
            statuses = [metric.status for metric in self.health_metrics.values()]
            if HealthStatus.CRITICAL in statuses:
                overall_status = HealthStatus.CRITICAL
            elif HealthStatus.WARNING in statuses:
                overall_status = HealthStatus.WARNING
            else:
                overall_status = HealthStatus.HEALTHY
        
        # 统计告警
        active_alerts = [alert for alert in self.alerts if not alert.resolved]
        warning_count = len([alert for alert in active_alerts if alert.level == "WARNING"])
        critical_count = len([alert for alert in active_alerts if alert.level == "CRITICAL"])
        
        return {
            'overall_status': overall_status.value,
            'timestamp': datetime.now().isoformat(),
            'metrics': {
                name: {
                    'value': metric.value,
                    'unit': metric.unit,
                    'status': metric.status.value,
                    'timestamp': metric.timestamp.isoformat()
                }
                for name, metric in self.health_metrics.items()
            },
            'alerts': {
                'total': len(active_alerts),
                'warning': warning_count,
                'critical': critical_count,
                'recent': [
                    {
                        'level': alert.level,
                        'component': alert.component,
                        'message': alert.message,
                        'timestamp': alert.timestamp.isoformat()
                    }
                    for alert in active_alerts[-10:]  # 最近10个告警
                ]
            }
        }
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """获取性能摘要"""
        if not self.performance_history:
            return {'message': '暂无性能数据'}
        
        # 计算平均值
        cpu_values = [record.get('cpu_usage', 0) for record in self.performance_history]
        memory_values = [record.get('memory_usage', 0) for record in self.performance_history]
        
        return {
            'period': f"{len(self.performance_history)} 个检查周期",
            'average_cpu': f"{sum(cpu_values) / len(cpu_values):.2f}%",
            'average_memory': f"{sum(memory_values) / len(memory_values):.2f}%",
            'max_cpu': f"{max(cpu_values):.2f}%",
            'max_memory': f"{max(memory_values):.2f}%"
        }
    
    def resolve_alert(self, component: str, message: str = None):
        """解决告警"""
        for alert in self.alerts:
            if alert.component == component and not alert.resolved:
                if message is None or message in alert.message:
                    alert.resolved = True
                    logger.info(f"告警已解决: {alert.message}")
                    break
    
    def clear_all_alerts(self):
        """清除所有告警"""
        for alert in self.alerts:
            alert.resolved = True
        logger.info("所有告警已清除")

# =============================================================================
# 程序入口
# =============================================================================
if __name__ == "__main__":
    # 测试系统监控器
    monitor = SystemMonitor(check_interval=5)
    
    # 注册恢复策略
    def cpu_recovery(status):
        print(f"CPU恢复策略执行: {status}")
    
    def memory_recovery(status):
        print(f"内存恢复策略执行: {status}")
    
    monitor.register_recovery_strategy('cpu_usage', cpu_recovery)
    monitor.register_recovery_strategy('memory_usage', memory_recovery)
    
    try:
        # 启动监控
        monitor.start_monitoring()
        
        # 运行一段时间
        time.sleep(30)
        
        # 获取健康状态
        health = monitor.get_health_status()
        print(f"系统健康状态: {health}")
        
        # 获取性能摘要
        performance = monitor.get_performance_summary()
        print(f"性能摘要: {performance}")
        
    finally:
        # 停止监控
        monitor.stop_monitoring()

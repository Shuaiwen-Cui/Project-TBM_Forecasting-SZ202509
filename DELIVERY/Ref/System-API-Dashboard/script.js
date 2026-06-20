// TBM盾构机实时监控系统 - 前端JavaScript

class TBMDataMonitor {
    constructor() {
        this.isRunning = false;
        this.updateInterval = null;
        this.lastData = null;
        this.startTime = new Date();
        this.paused = false;
        
        this.init();
    }

    // 初始化系统
    init() {
        this.setupEventListeners();
        this.renderDataTable();
        this.startMonitoring();
        this.updateStatus('connecting');
    }

    // 设置事件监听器
    setupEventListeners() {
        // 暂停/继续按钮
        document.getElementById('pauseBtn').addEventListener('click', () => {
            this.togglePause();
        });

        // 刷新按钮
        document.getElementById('refreshBtn').addEventListener('click', () => {
            this.refreshData();
        });

        // 键盘快捷键
        document.addEventListener('keydown', (e) => {
            if (e.key === ' ') {
                e.preventDefault();
                this.togglePause();
            } else if (e.key === 'r' || e.key === 'R') {
                this.refreshData();
            }
        });
    }

    // 渲染数据表格
    renderDataTable() {
        const tbody = document.getElementById('dataTableBody');
        tbody.innerHTML = '';

        FEATURE_CONFIG.forEach(feature => {
            const row = document.createElement('tr');
            row.innerHTML = `
                <td>${feature.id}</td>
                <td>${feature.name} (${feature.unit})</td>
                <td class="data-value" id="value-${feature.id}">--</td>
                <td class="data-value" id="prediction-${feature.id}">--</td>
            `;
            tbody.appendChild(row);
        });
    }

    // 开始监控
    startMonitoring() {
        this.isRunning = true;
        this.updateInterval = setInterval(() => {
            if (!this.paused) {
                this.fetchData();
            }
        }, 2000); // 每2秒更新一次

        // 立即获取一次数据
        this.fetchData();
    }

    // 停止监控
    stopMonitoring() {
        this.isRunning = false;
        if (this.updateInterval) {
            clearInterval(this.updateInterval);
            this.updateInterval = null;
        }
    }

    // 切换暂停状态
    togglePause() {
        this.paused = !this.paused;
        const pauseBtn = document.getElementById('pauseBtn');
        const icon = pauseBtn.querySelector('i');
        
        if (this.paused) {
            icon.className = 'fas fa-play';
            pauseBtn.innerHTML = '<i class="fas fa-play"></i> 继续';
            pauseBtn.classList.remove('btn-secondary');
            pauseBtn.classList.add('btn-primary');
        } else {
            icon.className = 'fas fa-pause';
            pauseBtn.innerHTML = '<i class="fas fa-pause"></i> 暂停';
            pauseBtn.classList.remove('btn-primary');
            pauseBtn.classList.add('btn-secondary');
        }
    }

    // 刷新数据
    refreshData() {
        this.showLoading();
        this.fetchData();
    }

    // 获取数据
    async fetchData() {
        try {
            console.log('正在尝试获取数据...');
            const response = await fetch('/api/tbm-data', {
                method: 'GET',
                headers: {
                    'Content-Type': 'application/json',
                },
                mode: 'cors'
            });
            
            console.log('API响应状态:', response.status);
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const data = await response.json();
            console.log('获取到数据:', data);
            this.updateData(data);
            this.updateStatus('connected');
            this.hideLoading();
        } catch (error) {
            console.error('获取数据失败:', error);
            this.updateStatus('error');
            this.hideLoading();
            
            // 如果API失败，使用模拟数据
            console.log('使用模拟数据...');
            this.updateData(this.generateMockData());
        }
    }

    // 更新数据
    updateData(data) {
        if (!data || !data.features) {
            console.warn('无效的数据格式');
            return;
        }

        const features = data.features;
        let validCount = 0;
        let missingCount = 0;
        let predictedCount = 0;

        FEATURE_CONFIG.forEach(feature => {
            const currentValue = features[feature.id - 1];
            // 直接使用后端返回的数据，不做任何处理
            const currentStatus = this.determineDataStatus(currentValue);
            
            // 预测值总是通过模型生成，不管当前值是否存在
            const prediction = this.generateModelPrediction(feature.id, currentValue);
            
            this.updateFeatureDisplay(feature.id, currentValue, currentStatus, prediction);
            
            // 统计数据 - 根据状态分类
            if (currentStatus === 'valid') {
                validCount++;
            } else if (currentStatus === 'predicted') {
                predictedCount++;
            } else if (currentStatus === 'missing') {
                missingCount++;
            }
        });

        // 更新统计信息
        this.updateStats(validCount, missingCount, predictedCount);
        
        // 更新运行时间
        this.updateRunTime();
        
        // 保存当前数据用于趋势计算
        this.lastData = features;
    }



    // 确定数据状态
    determineDataStatus(value) {
        if (value === null || value === undefined) {
            return 'missing';
        }
        
        // 检查是否是填充数据（对象且标记为predicted）
        if (typeof value === 'object' && value.predicted === true) {
            return 'predicted';
        }
        
        // 检查是否是有效数据
        if (typeof value === 'number' && !isNaN(value)) {
            return 'valid';
        }
        
        // 检查是否是对象但非填充数据
        if (typeof value === 'object' && value.value !== undefined) {
            return 'valid';
        }
        
        return 'error';
    }

    // 生成模型预测值（总是有结果）
    generateModelPrediction(featureId, currentValue) {
        // 预测值总是通过模型生成，不管当前值是否存在
        // 这里模拟AI模型的预测逻辑
        
        let baseValue;
        
        if (currentValue !== null && currentValue !== undefined) {
            // 如果有当前值，基于当前值预测
            baseValue = typeof currentValue === 'object' ? currentValue.value : currentValue;
        } else {
            // 如果没有当前值，使用历史数据或默认值进行预测
            baseValue = this.getHistoricalValue(featureId) || this.getDefaultValue(featureId);
        }
        
        // 模拟AI模型的预测逻辑
        // 添加趋势预测和随机变化
        const trendFactor = this.calculateTrendFactor(featureId);
        const randomFactor = 0.05 * (Math.random() - 0.5) * 2; // ±5%随机变化
        const prediction = baseValue * (1 + trendFactor + randomFactor);
        
        // 确保预测值在合理范围内
        const minValue = Math.max(0, baseValue * 0.5);
        const maxValue = baseValue * 1.5;
        
        return Math.max(minValue, Math.min(maxValue, prediction));
    }
    
    // 获取历史值（用于预测）
    getHistoricalValue(featureId) {
        if (this.lastData && this.lastData[featureId - 1] !== null) {
            const lastValue = this.lastData[featureId - 1];
            return typeof lastValue === 'object' ? lastValue.value : lastValue;
        }
        return null;
    }
    
    // 获取默认值（用于预测）
    getDefaultValue(featureId) {
        // 根据特征类型返回合理的默认值
        const feature = FEATURE_CONFIG[featureId - 1];
        switch (feature.unit) {
            case 'MPa':
                return 10.0;
            case 'mm/min':
                return 50.0;
            case 'mm':
                return 500.0;
            case 'mm/rpm':
                return 15.0;  // 贯入度默认值
            case 'kN':
                return 5000.0;
            case 'r/min':
                return 2.0;
            case 'kN·m':
                return 500.0;
            case '%':
                return 50.0;
            default:
                return 50.0;
        }
    }
    
    // 计算趋势因子
    calculateTrendFactor(featureId) {
        if (!this.lastData) {
            return 0;
        }
        
        const currentValue = this.lastData[featureId - 1];
        if (currentValue === null || currentValue === undefined) {
            return 0;
        }
        
        const actualValue = typeof currentValue === 'object' ? currentValue.value : currentValue;
        
        // 简单的趋势计算：基于历史数据变化
        const trend = Math.sin(Date.now() / 10000) * 0.1; // 模拟周期性趋势
        return trend;
    }

    // 更新特征显示
    updateFeatureDisplay(featureId, value, status, prediction) {
        const valueCell = document.getElementById(`value-${featureId}`);
        const predictionCell = document.getElementById(`prediction-${featureId}`);

        // 更新当前值 - 使用统一的status参数
        if (value !== null && value !== undefined) {
            const displayValue = typeof value === 'object' ? value.value : value;
            valueCell.textContent = this.formatValue(displayValue);
            
            // 根据status设置颜色
            // 由于API数据通常是历史数据，大部分当前值都是填充的，显示橙色
            if (status === 'predicted') {
                valueCell.className = `data-value missing`; // 橙色 - 填充数据（主要情况）
            } else if (status === 'valid') {
                // 只有在极少数情况下才有实时数据，显示绿色
                valueCell.className = `data-value valid`; // 绿色 - 实时数据（罕见）
            } else {
                valueCell.className = `data-value missing`; // 橙色 - 其他情况
            }
        } else {
            valueCell.textContent = '--';
            valueCell.className = 'data-value missing'; // 橙色 - 缺失数据
        }

        // 更新预测值（预测值总是存在）
        const displayPrediction = typeof prediction === 'object' ? prediction.value : prediction;
        predictionCell.textContent = this.formatValue(displayPrediction);
        predictionCell.className = `data-value predicted`;
    }

    // 格式化数值
    formatValue(value) {
        if (typeof value !== 'number' || isNaN(value)) {
            return '--';
        }
        
        if (value >= 1000) {
            return value.toFixed(0);
        } else if (value >= 1) {
            return value.toFixed(2);
        } else {
            return value.toFixed(4);
        }
    }

    // 更新统计信息
    updateStats(validCount, missingCount, predictedCount) {
        document.getElementById('validCount').textContent = validCount;
        document.getElementById('missingCount').textContent = missingCount;
        // 预测值总是31个，不需要更新
    }

    // 更新运行时间
    updateRunTime() {
        const now = new Date();
        const diff = now - this.startTime;
        const hours = Math.floor(diff / 3600000);
        const minutes = Math.floor((diff % 3600000) / 60000);
        const seconds = Math.floor((diff % 60000) / 1000);
        
        const timeString = `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
        document.getElementById('runTime').textContent = timeString;
    }

    // 更新连接状态
    updateStatus(status) {
        const statusDot = document.getElementById('statusDot');
        const statusText = document.getElementById('statusText');
        const lastUpdate = document.getElementById('lastUpdate');

        statusDot.className = `status-dot ${status}`;
        
        switch (status) {
            case 'connected':
                statusText.textContent = '已连接';
                lastUpdate.textContent = `最后更新: ${new Date().toLocaleTimeString()}`;
                break;
            case 'connecting':
                statusText.textContent = '连接中...';
                lastUpdate.textContent = '';
                break;
            case 'error':
                statusText.textContent = '连接错误';
                lastUpdate.textContent = `错误时间: ${new Date().toLocaleTimeString()}`;
                break;
        }
    }

    // 显示加载动画
    showLoading() {
        document.getElementById('loadingOverlay').classList.add('show');
    }

    // 隐藏加载动画
    hideLoading() {
        document.getElementById('loadingOverlay').classList.remove('show');
    }

    // 生成模拟数据（用于测试）
    generateMockData() {
        const features = [];
        
        FEATURE_CONFIG.forEach((feature, index) => {
            // 模拟数据：80%概率有有效数据，20%概率缺失
            if (Math.random() < 0.8) {
                // 生成合理范围内的随机值
                let value;
                switch (feature.unit) {
                    case 'MPa':
                        value = Math.random() * 20;
                        break;
                    case 'mm/min':
                        value = Math.random() * 100;
                        break;
                    case 'mm':
                        value = Math.random() * 1000;
                        break;
                    case 'kN':
                        value = Math.random() * 10000;
                        break;
                    case 'r/min':
                        value = Math.random() * 5;
                        break;
                    case 'kN·m':
                        value = Math.random() * 1000;
                        break;
                    case '%':
                        value = Math.random() * 100;
                        break;
                    default:
                        value = Math.random() * 100;
                }
                
                // 10%概率标记为预测值
                if (Math.random() < 0.1) {
                    features.push({ value: value, predicted: true });
                } else {
                    features.push(value);
                }
            } else {
                features.push(null);
            }
        });

        return { features: features };
    }
}

// 页面加载完成后初始化系统
document.addEventListener('DOMContentLoaded', () => {
    window.tbmMonitor = new TBMDataMonitor();
    
    // 添加页面可见性变化监听
    document.addEventListener('visibilitychange', () => {
        if (document.hidden) {
            console.log('页面隐藏，暂停监控');
        } else {
            console.log('页面显示，恢复监控');
        }
    });
});

// 页面卸载时清理资源
window.addEventListener('beforeunload', () => {
    if (window.tbmMonitor) {
        window.tbmMonitor.stopMonitoring();
    }
});

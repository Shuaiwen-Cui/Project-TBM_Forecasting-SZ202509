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
from api_client import TBMAPIClient  # 导入API客户端模块


# =============================================================================
# 系统配置模块
# =============================================================================
# 模型参数配置
FEATURE_NUM = 31        # 特征数量
INPUT_LENGTH = 5        # 输入时间步长度
UPDATE_INTERVAL = 60    # 更新间隔(秒) - 每分钟更新一次
DATA_FETCH_SECOND = 10  # 每分钟的第几秒拉取数据

# 数据获取模式配置
DATA_MODE = 3           # 数据获取模式: 1=随机生成, 2=API获取+0填充, 3=API获取+随机填充, 4=API获取+预测值填充

# 数据记录配置
ENABLE_DATA_LOGGING = True    # 是否启用数据记录功能
LOG_DATA_TO_FILE = True       # 是否记录数据到文件
LOG_PREDICTIONS_TO_FILE = True # 是否记录预测到文件

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
        self.api_client = TBMAPIClient()                 # API客户端实例
        self.buffer = np.zeros((INPUT_LENGTH, FEATURE_NUM))  # 滑动窗口缓冲区
        self.step_count = 0                              # 步数计数器
        self.last_data = None                            # 上次生成的数据
        self.last_prediction = None                      # 上次预测结果
        self.last_api_data = None                        # 上次API获取的数据
        self.data_mode = DATA_MODE                       # 数据获取模式
        self.buffer_initialized = False                  # 缓冲区是否已初始化
        
        # 数据记录功能
        self.data_log_file = None                        # 数据记录文件
        self.prediction_log_file = None                  # 预测记录文件
        self.enable_logging = ENABLE_DATA_LOGGING        # 是否启用数据记录
        self.log_data = LOG_DATA_TO_FILE                 # 是否记录数据
        self.log_predictions = LOG_PREDICTIONS_TO_FILE   # 是否记录预测
        
        if self.enable_logging:
            self._init_data_logging()                    # 初始化数据记录
    
    # =========================================================================
    # 数据记录模块
    # =========================================================================
    def _init_data_logging(self):
        """初始化数据记录功能"""
        if not self.enable_logging:
            print("📝 数据记录功能已禁用")
            return
            
        try:
            # 创建带时间戳的文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # 根据配置创建文件
            if self.log_data:
                self.data_log_file = f"tbm_data_{timestamp}.txt"
                with open(self.data_log_file, 'w', encoding='utf-8') as f:
                    # 写入表头
                    header = "时间戳\t步数\t" + "\t".join([f"特征{i+1}_{name}" for i, name in enumerate(FEATURE_NAMES)])
                    f.write(header + "\n")
            
            if self.log_predictions:
                self.prediction_log_file = f"tbm_predictions_{timestamp}.txt"
                with open(self.prediction_log_file, 'w', encoding='utf-8') as f:
                    # 写入表头
                    header = "时间戳\t步数\t" + "\t".join([f"预测{i+1}_{name}" for i, name in enumerate(FEATURE_NAMES)])
                    f.write(header + "\n")
            
            print(f"📝 数据记录配置:")
            print(f"   启用记录: {self.enable_logging}")
            print(f"   记录数据: {self.log_data}")
            print(f"   记录预测: {self.log_predictions}")
            if self.data_log_file:
                print(f"   数据文件: {self.data_log_file}")
            if self.prediction_log_file:
                print(f"   预测文件: {self.prediction_log_file}")
            
        except Exception as e:
            print(f"❌ 数据记录初始化失败: {e}")
            self.data_log_file = None
            self.prediction_log_file = None
    
    def _log_data(self, data, step_count, data_type="current"):
        """
        记录数据到文件
        
        Args:
            data: 数据数组
            step_count: 步数
            data_type: 数据类型 ("current" 或 "prediction")
        """
        # 检查是否启用记录功能
        if not self.enable_logging:
            return
        
        # 检查具体记录类型
        if data_type == "current" and not self.log_data:
            return
        if data_type == "prediction" and not self.log_predictions:
            return
        
        # 检查文件是否存在
        if data_type == "current" and not self.data_log_file:
            return
        if data_type == "prediction" and not self.prediction_log_file:
            return
        
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            if data_type == "current" and self.data_log_file:
                # 记录当前数据
                with open(self.data_log_file, 'a', encoding='utf-8') as f:
                    data_str = "\t".join([f"{val:.6f}" if val is not None else "None" for val in data])
                    f.write(f"{timestamp}\t{step_count}\t{data_str}\n")
            
            elif data_type == "prediction" and self.prediction_log_file:
                # 记录预测数据
                with open(self.prediction_log_file, 'a', encoding='utf-8') as f:
                    data_str = "\t".join([f"{val:.6f}" if val is not None else "None" for val in data])
                    f.write(f"{timestamp}\t{step_count}\t{data_str}\n")
                    
        except Exception as e:
            print(f"⚠️  数据记录失败: {e}")
    
    def _close_data_logging(self):
        """关闭数据记录"""
        if not self.enable_logging:
            return
            
        try:
            if self.data_log_file and self.log_data:
                print(f"📝 数据记录已保存到: {self.data_log_file}")
            if self.prediction_log_file and self.log_predictions:
                print(f"📝 预测记录已保存到: {self.prediction_log_file}")
        except Exception as e:
            print(f"⚠️  关闭数据记录时出错: {e}")
    
    def set_logging_config(self, enable_logging=None, log_data=None, log_predictions=None):
        """
        动态设置记录配置
        
        Args:
            enable_logging (bool): 是否启用记录功能
            log_data (bool): 是否记录数据
            log_predictions (bool): 是否记录预测
        """
        if enable_logging is not None:
            self.enable_logging = enable_logging
        if log_data is not None:
            self.log_data = log_data
        if log_predictions is not None:
            self.log_predictions = log_predictions
        
        print(f"📝 记录配置已更新:")
        print(f"   启用记录: {self.enable_logging}")
        print(f"   记录数据: {self.log_data}")
        print(f"   记录预测: {self.log_predictions}")
    
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
    
    def test_api_connection(self):
        """
        测试API连接
        
        Returns:
            bool: 连接成功返回True，失败返回False
        """
        return self.api_client.test_connection()
    
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
        
        # 获取5分钟历史数据
        historical_data = self.api_client.get_historical_data(minutes_back=5)
        
        if not historical_data:
            print("❌ 无法获取历史数据，使用模拟数据初始化")
            # 使用模拟数据初始化
            for i in range(INPUT_LENGTH):
                simulated_data = self._generate_simulated_data()
                self.buffer[i] = simulated_data
                self.step_count += 1
                print(f"   步骤 {i+1}: 使用模拟数据")
        else:
            # 使用历史数据初始化
            print(f"📊 使用 {len(historical_data)} 条历史记录初始化缓冲区")
            
            # 选择最近的5条记录（如果历史数据超过5条）
            if len(historical_data) >= INPUT_LENGTH:
                selected_data = historical_data[-INPUT_LENGTH:]
            else:
                # 如果历史数据不足5条，用模拟数据补充
                selected_data = historical_data.copy()
                while len(selected_data) < INPUT_LENGTH:
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
                filled_features = self.fill_missing_data(features, fill_mode=3)
                self.buffer[i] = filled_features
                self.step_count += 1
                
                # 记录历史数据
                self._log_data(filled_features, self.step_count, "current")
                
                print(f"   步骤 {i+1}: ID={record_id}, 时间={timestamp}, 有效特征={valid_count}/31")
        
        self.buffer_initialized = True
        print("=" * 60)
        print(f"✅ 缓冲区初始化完成！步数: {self.step_count}/{INPUT_LENGTH}")
        print("=" * 60)
        return True
    
    def _generate_simulated_data(self):
        """生成模拟数据"""
        data = []
        for i in range(FEATURE_NUM):
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
    
    def set_data_mode(self, mode):
        """
        设置数据获取模式
        
        Args:
            mode (int): 数据模式 1-4
        """
        if mode in [1, 2, 3, 4]:
            self.data_mode = mode
            mode_names = {
                1: "随机生成数据",
                2: "API获取数据 + 0填充缺失值",
                3: "API获取数据 + 随机值填充缺失值",
                4: "API获取数据 + 预测值填充缺失值"
            }
            print(f"✅ 数据模式已切换为: {mode} - {mode_names[mode]}")
        else:
            print(f"❌ 无效的数据模式: {mode}，请使用1-4")
    
    # =========================================================================
    # 数据管理模块
    # =========================================================================
    def fetch_api_data(self):
        """
        从API获取真实数据
        
        Returns:
            np.ndarray: API获取的数据，缺失值用None表示
        """
        print("📡 正在从API获取数据...")
        
        try:
            # 使用API客户端获取最新特征数据
            api_data = self.api_client.get_latest_features()
            
            # 检查数据是否与上一次相同
            if self.last_api_data is not None:
                if self._is_data_same(api_data, self.last_api_data):
                    print("⚠️  检测到API数据与上一次相同，认为没有获取到新数据")
                    # 返回全None数组表示没有新数据
                    return np.full(FEATURE_NUM, None)
            
            # 保存当前数据作为下次比较的基准
            self.last_api_data = api_data.copy() if api_data is not None else None
            return api_data
            
        except Exception as e:
            print(f"❌ API数据获取失败: {e}")
            # 返回全None数组表示获取失败
            return np.full(FEATURE_NUM, None)
    
    def _is_data_same(self, data1, data2):
        """
        比较两次API数据是否相同
        
        Args:
            data1 (np.ndarray): 当前API数据
            data2 (np.ndarray): 上次API数据
            
        Returns:
            bool: 如果数据相同返回True，否则返回False
        """
        if data1 is None and data2 is None:
            return True
        if data1 is None or data2 is None:
            return False
        
        # 比较有效数据（非None值）
        for i in range(FEATURE_NUM):
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
    
    def fill_missing_data(self, api_data, fill_mode):
        """
        填充缺失数据
        
        Args:
            api_data (np.ndarray): API获取的数据
            fill_mode (int): 填充模式
            
        Returns:
            np.ndarray: 填充后的完整数据
        """
        filled_data = api_data.copy()
        
        for i in range(FEATURE_NUM):
            if api_data[i] is None:  # 数据缺失
                if fill_mode == 2:  # 用0填充
                    filled_data[i] = 0.0
                    print(f"  ⚠️  特征{i+1}缺失，用0填充")
                    
                elif fill_mode == 3:  # 用随机值填充
                    if self.model.is_loaded:
                        data_min, data_max = self.model.get_data_range()
                        filled_data[i] = random.uniform(data_min[i], data_max[i])
                    else:
                        filled_data[i] = random.uniform(0, 100)
                    print(f"  ⚠️  特征{i+1}缺失，用随机值填充")
                    
                elif fill_mode == 4:  # 用预测值填充
                    if self.last_prediction is not None:
                        filled_data[i] = self.last_prediction[i]
                        print(f"  ⚠️  特征{i+1}缺失，用预测值填充")
                    else:
                        # 没有预测值时用随机值
                        if self.model.is_loaded:
                            data_min, data_max = self.model.get_data_range()
                            filled_data[i] = random.uniform(data_min[i], data_max[i])
                        else:
                            filled_data[i] = random.uniform(0, 100)
                        print(f"  ⚠️  特征{i+1}缺失，用随机值填充（无预测值）")
        
        return filled_data
    
    def generate_data(self):
        """
        生成数据（根据配置模式）
        
        数据生成策略:
        1. 随机生成数据
        2. API获取数据 + 0填充缺失值
        3. API获取数据 + 随机值填充缺失值
        4. API获取数据 + 预测值填充缺失值
        
        Returns:
            np.ndarray: 生成的31维特征数据
        """
        if self.data_mode == 1:
            # 模式1: 随机生成数据
            return self._generate_random_data()
        else:
            # 模式2-4: API获取数据
            api_data = self.fetch_api_data()
            return self.fill_missing_data(api_data, self.data_mode)
    
    def _generate_random_data(self):
        """
        生成随机数据（模式1）
        
        Returns:
            np.ndarray: 随机生成的31维特征数据
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
        
        # 添加新数据 - 处理None值
        if new_data is not None:
            # 将None值替换为0，避免nan问题
            processed_data = []
            for val in new_data:
                if val is None:
                    processed_data.append(0.0)
                else:
                    processed_data.append(float(val))
            self.buffer[-1] = processed_data
        else:
            # 如果新数据为None，用0填充
            self.buffer[-1] = [0.0] * FEATURE_NUM
        self.step_count += 1
        
        # 记录当前数据
        self._log_data(new_data, self.step_count, "current")
    
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
        prediction = self.model.predict(input_data)
        self.last_prediction = prediction.copy()  # 保存预测结果用于填充
        
        # 记录预测数据
        self._log_data(prediction, self.step_count, "prediction")
        
        return prediction
    
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
        
        # 显示数据模式
        mode_names = {
            1: "随机生成数据",
            2: "API获取数据 + 0填充缺失值",
            3: "API获取数据 + 随机值填充缺失值",
            4: "API获取数据 + 预测值填充缺失值"
        }
        print(f"数据模式: {self.data_mode} - {mode_names.get(self.data_mode, '未知模式')}")
        
        # 步骤3: 测试API连接（如果使用API模式）
        if self.data_mode > 1:
            print("\n🔍 测试API连接...")
            if not self.test_api_connection():
                print("⚠️  API连接失败，将使用随机数据模式")
                self.data_mode = 1
            else:
                print("✅ API连接成功")
        
        # 步骤4: 初始化缓冲区（拉取5分钟历史数据）
        print("\n🔄 初始化缓冲区...")
        if not self.initialize_buffer():
            print("❌ 缓冲区初始化失败，程序退出")
            return
        
        try:
            # 步骤5: 主循环
            while True:
                current_time = datetime.now()
                current_second = current_time.second
                
                # 检查是否到了每分钟的第10秒
                if current_second == DATA_FETCH_SECOND:
                    print(f"\n🕐 时间: {current_time.strftime('%H:%M:%S')} - 开始拉取数据")
                    
                    # 5.1 生成新数据
                    new_data = self.generate_data()
                    
                    # 5.2 更新滑动窗口（挤掉最早的数据）
                    self.update_buffer(new_data)
                    
                    # 5.3 执行预测（缓冲区已初始化，可以直接预测）
                    prediction = self.predict(self.buffer)
                    self.print_prediction(prediction, self.step_count - INPUT_LENGTH + 1)
                
                # 3.4 等待1秒后检查时间
                time.sleep(1)
                
        except KeyboardInterrupt:
            # 步骤4: 处理中断信号
            print(f"\n程序结束 - 总步数: {self.step_count}")
            self._close_data_logging()


# =============================================================================
# 程序入口
# =============================================================================
if __name__ == "__main__":
    # 创建预测系统实例并运行
    predictor = TBMPredictor()
    predictor.run()
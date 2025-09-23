#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TBM预测模块 (predict.py)
========================

模块功能:
- 模型加载和管理
- 数据预处理和标准化
- 模型推理和预测
- 数据范围获取

模块职责:
- 封装ONNX模型相关操作
- 提供统一的预测接口
- 处理数据标准化/反标准化
"""

# =============================================================================
# 导入依赖库
# =============================================================================
import onnxruntime as ort
import numpy as np
import joblib
import os
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# 模型预测器类
# =============================================================================
class ModelPredictor:
    """
    TBM模型预测器
    
    职责:
    1. 加载和管理ONNX模型
    2. 处理数据标准化/反标准化
    3. 执行模型推理
    4. 提供数据范围信息
    """
    
    def __init__(self):
        """初始化预测器"""
        self.session = None      # ONNX推理会话
        self.scaler = None       # 数据标准化器
        self.is_loaded = False   # 模型加载状态
    
    # =========================================================================
    # 模型管理模块
    # =========================================================================
    def load_model(self, model_dir=None):
        """
        加载ONNX模型和归一化器
        
        Args:
            model_dir (str, optional): 模型文件目录路径
            
        Returns:
            bool: 加载成功返回True，失败返回False
        """
        # 设置默认模型目录
        if model_dir is None:
            model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'AI-Model')
        
        try:
            # 构建文件路径
            model_path = os.path.join(model_dir, 'model_transformer.onnx')
            scaler_path = os.path.join(model_dir, 'minmax_scaler.pkl')
            
            # 检查文件是否存在
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"模型文件不存在: {model_path}")
            if not os.path.exists(scaler_path):
                raise FileNotFoundError(f"归一化器文件不存在: {scaler_path}")
            
            # 加载模型和归一化器
            self.session = ort.InferenceSession(model_path)
            self.scaler = joblib.load(scaler_path)
            self.is_loaded = True
            
            print("✓ 模型加载成功")
            return True
            
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            self.is_loaded = False
            return False
    
    # =========================================================================
    # 预测执行模块
    # =========================================================================
    def predict(self, input_data):
        """
        执行模型预测
        
        Args:
            input_data (np.ndarray): 输入数据，形状为 (时间步, 特征数)
            
        Returns:
            np.ndarray: 预测结果，形状为 (特征数,)
            
        Raises:
            RuntimeError: 模型未加载时抛出异常
        """
        if not self.is_loaded:
            raise RuntimeError("模型未加载")
        
        # 步骤1: 数据标准化
        input_scaled = self.scaler.transform(
            input_data.reshape(-1, input_data.shape[-1])
        ).reshape(1, input_data.shape[0], input_data.shape[1]).astype(np.float32)
        
        # 步骤2: 模型推理
        output_scaled = self.session.run(
            None, {self.session.get_inputs()[0].name: input_scaled}
        )
        
        # 步骤3: 反标准化
        prediction = self.scaler.inverse_transform(output_scaled[0])
        
        return prediction[0]
    
    # =========================================================================
    # 数据信息模块
    # =========================================================================
    def get_data_range(self):
        """
        获取数据范围信息
        
        Returns:
            tuple: (data_min, data_max) 数据最小值和最大值
        """
        if not self.is_loaded:
            return None, None
        return self.scaler.data_min_, self.scaler.data_max_

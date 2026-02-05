"""
数据加载和预处理模块
支持不同时间尺度的序列生成，便于多时间尺度对比实验
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path
import config


def load_data(data_file):
    """
    加载预处理后的数据
    
    Args:
        data_file: 数据文件路径
    
    Returns:
        pd.DataFrame: 数据框
    """
    if not Path(data_file).exists():
        raise FileNotFoundError(f"数据文件不存在: {data_file}")
    
    # 使用low_memory=False避免类型警告
    df = pd.read_csv(data_file, encoding='utf-8-sig', low_memory=False)
    
    # 跳过前两行（标题和单位行，如果有）
    if len(df) > 2:
        df = df.iloc[2:].reset_index(drop=True)
    
    # 转换为数值类型（前N列通常是元数据）
    numeric_cols = df.columns[config.METADATA_COLS:] if len(df.columns) > config.METADATA_COLS else df.columns
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # 填充缺失值
    df[numeric_cols] = df[numeric_cols].ffill().bfill()
    
    return df


def prepare_sequences(data, seq_len, pred_len, batch_size=10000):
    """
    准备时间序列数据（分批处理以节省内存）
    
    Args:
        data: 数据数组 (n_samples, n_features)
        seq_len: 输入序列长度
        pred_len: 预测长度
        batch_size: 分批处理的大小（对于大数据集）
    
    Returns:
        X: 输入序列 (n_samples, seq_len, n_features)
        Y: 输出序列 (n_samples, pred_len, n_features)
    """
    max_start = len(data) - seq_len - pred_len + 1
    
    # 对于大数据集，分批处理
    if max_start > batch_size:
        X_list, Y_list = [], []
        for start_idx in range(0, max_start, batch_size):
            end_idx = min(start_idx + batch_size, max_start)
            batch_X, batch_Y = [], []
            
            for i in range(start_idx, end_idx):
                batch_X.append(data[i:i+seq_len])
                batch_Y.append(data[i+seq_len:i+seq_len+pred_len])
            
            X_list.append(np.array(batch_X, dtype=np.float32))
            Y_list.append(np.array(batch_Y, dtype=np.float32))
        
        # 合并所有批次
        X = np.concatenate(X_list, axis=0)
        Y = np.concatenate(Y_list, axis=0)
        
        # 清理中间变量
        del X_list, Y_list
    else:
        # 小数据集直接处理
        X, Y = [], []
        for i in range(max_start):
            X.append(data[i:i+seq_len])
            Y.append(data[i+seq_len:i+seq_len+pred_len])
        
        X = np.array(X, dtype=np.float32)
        Y = np.array(Y, dtype=np.float32)
    
    return X, Y


def get_feature_names_from_config(n_features):
    """
    从配置中获取特征名称列表
    
    Args:
        n_features: 特征数量
    
    Returns:
        list: 特征名称列表（中文）
    """
    feature_names = []
    for i in range(n_features):
        name = config.get_feature_name(i, lang='cn')
        feature_names.append(name)
    return feature_names


def prepare_data_for_experiment(data_file, seq_len, pred_len, 
                                feature_start_idx=None,
                                random_seed=42):
    """
    为实验准备数据（包括归一化和划分）
    每次只加载当前实验需要的数据，用完即释放
    
    Args:
        data_file: 数据文件路径
        seq_len: 序列长度
        pred_len: 预测长度
        feature_start_idx: 特征列起始索引（None则使用config.METADATA_COLS）
        random_seed: 随机种子
    
    Returns:
        dict: 包含训练/验证/测试数据和scaler的字典
    """
    # 使用config中的配置
    if feature_start_idx is None:
        feature_start_idx = config.METADATA_COLS
    
    # 加载数据
    df = load_data(data_file)
    
    # 提取特征（跳过前N列元数据）
    feature_data = df.iloc[:, feature_start_idx:].values.astype(np.float32)
    
    # 获取特征名称（优先使用config中的映射，否则使用原始列名）
    n_features = feature_data.shape[1]
    if n_features <= len(config.FEATURE_NAME_MAPPING):
        # 使用config中的特征名称映射
        feature_names = get_feature_names_from_config(n_features)
    else:
        # 如果特征数量超过映射，使用原始列名
        feature_names = df.columns[feature_start_idx:].tolist()
        print(f"警告: 特征数量({n_features})超过配置中的映射，使用原始列名")
    
    # 归一化
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(feature_data)
    
    # 确保是float32以节省内存
    data_scaled = data_scaled.astype(np.float32)
    
    # 准备序列
    X, Y = prepare_sequences(data_scaled, seq_len, pred_len)
    
    # 确保是float32
    X = X.astype(np.float32)
    Y = Y.astype(np.float32)
    
    # 数据划分（时间序列不打乱）
    n_samples = len(X)
    n_train = int(n_samples * config.TRAIN_RATIO)
    n_val = int(n_samples * config.VAL_RATIO)
    
    X_train = X[:n_train]
    Y_train = Y[:n_train]
    X_val = X[n_train:n_train+n_val]
    Y_val = Y[n_train:n_train+n_val]
    X_test = X[n_train+n_val:]
    Y_test = Y[n_train+n_val:]
    
    # 清理中间变量
    del df, feature_data, data_scaled, X, Y
    
    return {
        'X_train': X_train,
        'Y_train': Y_train,
        'X_val': X_val,
        'Y_val': Y_val,
        'X_test': X_test,
        'Y_test': Y_test,
        'scaler': scaler,
        'feature_names': feature_names,
        'n_features': len(feature_names)
    }

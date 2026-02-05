"""
工具函数模块 - 评估指标、时间测量、结果保存
基于plan.txt的评估指标要求
"""
import time
import pickle
import psutil
import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import config


def calculate_metrics(y_true, y_pred):
    """
    计算评估指标（基于plan.txt的精度类指标）
    
    Args:
        y_true: 真实值 (n_samples, n_features) 或 (n_samples * pred_len, n_features)
        y_pred: 预测值 (n_samples, n_features) 或 (n_samples * pred_len, n_features)
    
    Returns:
        dict: 包含所有指标的字典
    """
    # 确保形状一致
    y_true = np.array(y_true).flatten() if y_true.ndim > 1 else np.array(y_true)
    y_pred = np.array(y_pred).flatten() if y_pred.ndim > 1 else np.array(y_pred)
    
    # 避免除零错误
    mask = y_true != 0
    y_true_masked = y_true[mask]
    y_pred_masked = y_pred[mask]
    
    # 精度类指标（基于plan.txt）
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    
    # MAPE (Mean Absolute Percentage Error)
    if len(y_true_masked) > 0:
        mape = np.mean(np.abs((y_true_masked - y_pred_masked) / y_true_masked)) * 100
    else:
        mape = np.nan
    
    # R²
    r2 = r2_score(y_true, y_pred)
    
    return {
        'MSE': float(mse),
        'MAE': float(mae),
        'RMSE': float(rmse),
        'MAPE': float(mape),
        'R2': float(r2)
    }


def measure_inference_time(model_wrapper, model, x_sample, pred_len, n_runs=10):
    """
    测量单样本推理时间（性能类指标）
    
    Args:
        model_wrapper: 模型包装器（实际上就是模型本身）
        model: 训练好的模型（与model_wrapper相同，为了兼容性保留）
        x_sample: 单个样本 (seq_len, n_features) 或 (1, seq_len, n_features)
        pred_len: 预测长度
        n_runs: 运行次数（取平均）
    
    Returns:
        float: 平均推理时间（毫秒）
    """
    # 确保x_sample是3D
    if x_sample.ndim == 2:
        x_sample = x_sample.reshape(1, *x_sample.shape)
    
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        _ = model_wrapper.predict(x_sample, pred_len=None)  # 模型已经知道pred_len，不需要传递
        end = time.perf_counter()
        times.append((end - start) * 1000)  # 转换为毫秒
    
    return np.mean(times)


def measure_batch_inference_time(model_wrapper, model, x_batch, pred_len, batch_size=100):
    """
    测量批量推理时间
    
    Args:
        model_wrapper: 模型包装器（实际上就是模型本身）
        model: 训练好的模型（与model_wrapper相同，为了兼容性保留）
        x_batch: 批量样本
        pred_len: 预测长度
        batch_size: 批量大小
    
    Returns:
        float: 批量推理时间（毫秒）
    """
    if len(x_batch) > batch_size:
        x_batch = x_batch[:batch_size]
    
    start = time.perf_counter()
    _ = model_wrapper.predict(x_batch, pred_len=None)  # 模型已经知道pred_len，不需要传递
    end = time.perf_counter()
    
    return (end - start) * 1000  # 转换为毫秒


def get_memory_usage():
    """
    获取当前进程的内存使用量（MB）
    
    Returns:
        float: 内存使用量（MB）
    """
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # 转换为MB


def count_model_parameters(model):
    """
    计算模型参数量（模型大小）
    
    Args:
        model: PyTorch模型或其他模型
    
    Returns:
        int: 参数量
    """
    try:
        import torch
        if isinstance(model, torch.nn.Module):
            return sum(p.numel() for p in model.parameters())
    except:
        pass
    
    # 对于非PyTorch模型（如ARIMA），返回估计值
    try:
        # ARIMA模型参数估计
        if hasattr(model, 'arparams') and hasattr(model, 'maparams'):
            return len(model.arparams) + len(model.maparams) + 1  # +1 for variance
    except:
        pass
    
    return 0


def save_experiment_result(result_dict, results_dir, model_name, seq_len, pred_len):
    """
    保存单个实验的详细结果
    
    Args:
        result_dict: 结果字典
        results_dir: 结果保存目录
        model_name: 模型名称
        seq_len: 序列长度
        pred_len: 预测长度
    
    Returns:
        str: 保存的文件路径
    """
    results_dir = Path(results_dir)
    results_dir.mkdir(exist_ok=True)
    
    filename = f"{model_name}_{seq_len}_{pred_len}_results.pkl"
    filepath = results_dir / filename
    
    with open(filepath, 'wb') as f:
        pickle.dump(result_dict, f)
    
    return str(filepath)


def load_experiment_result(results_dir, model_name, seq_len, pred_len):
    """
    加载单个实验的详细结果
    
    Args:
        results_dir: 结果保存目录
        model_name: 模型名称
        seq_len: 序列长度
        pred_len: 预测长度
    
    Returns:
        dict: 结果字典
    """
    results_dir = Path(results_dir)
    filename = f"{model_name}_{seq_len}_{pred_len}_results.pkl"
    filepath = results_dir / filename
    
    if not filepath.exists():
        return None
    
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def load_experiment_summary(results_dir):
    """
    加载实验汇总表
    
    Args:
        results_dir: 结果保存目录
    
    Returns:
        pd.DataFrame: 汇总表
    """
    results_dir = Path(results_dir)
    summary_file = results_dir / 'experiment_summary.csv'
    
    if summary_file.exists():
        return pd.read_csv(summary_file)
    else:
        # 创建空的汇总表（基于plan.txt的评估指标）
        columns = ['model', 'seq_len', 'pred_len', 
                  # 精度类指标
                  'MSE', 'MAE', 'RMSE', 'MAPE', 'R2',
                  # 模型大小
                  'model_size_params', 'model_size_mb',
                  # 性能类指标
                  'inference_time_ms', 'batch_inference_time_ms', 
                  'memory_usage_mb', 'training_time_s']
        return pd.DataFrame(columns=columns)


def update_experiment_summary(results_dir, result_dict):
    """
    更新实验汇总表（记录所有关键数据，便于论文写作）
    
    Args:
        results_dir: 结果保存目录
        result_dict: 结果字典
    """
    results_dir = Path(results_dir)
    summary_file = results_dir / 'experiment_summary.csv'
    
    # 加载现有汇总表
    summary_df = load_experiment_summary(results_dir)
    
    # 准备新行数据（基于plan.txt的评估指标）
    row = {
        'model': result_dict['model_name'],
        'seq_len': result_dict['config']['seq_len'],
        'pred_len': result_dict['config']['pred_len'],
        # 精度类指标
        'MSE': result_dict['metrics']['MSE'],
        'MAE': result_dict['metrics']['MAE'],
        'RMSE': result_dict['metrics']['RMSE'],
        'MAPE': result_dict['metrics']['MAPE'],
        'R2': result_dict['metrics']['R2'],
        # 模型大小
        'model_size_params': result_dict['costs']['model_size_params'],
        'model_size_mb': result_dict['costs']['model_size_mb'],
        # 性能类指标
        'inference_time_ms': result_dict['costs']['inference_time_ms'],
        'batch_inference_time_ms': result_dict['costs'].get('batch_inference_time_ms', np.nan),
        'memory_usage_mb': result_dict['costs']['memory_usage_mb'],
        'training_time_s': result_dict['costs']['training_time_s']
    }
    
    # 检查是否已存在（避免重复）
    if len(summary_df) > 0:
        mask = (
            (summary_df['model'] == row['model']) &
            (summary_df['seq_len'] == row['seq_len']) &
            (summary_df['pred_len'] == row['pred_len'])
        )
        if mask.any():
            # 更新现有行
            summary_df.loc[mask, list(row.keys())] = list(row.values())
        else:
            # 添加新行
            summary_df = pd.concat([summary_df, pd.DataFrame([row])], ignore_index=True)
    else:
        # 第一行
        summary_df = pd.DataFrame([row])
    
    # 保存
    summary_df.to_csv(summary_file, index=False)

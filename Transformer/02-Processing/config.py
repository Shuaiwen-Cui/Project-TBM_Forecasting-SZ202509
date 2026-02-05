"""
实验配置文件 - 基于plan.txt的实验设计
统一管理所有实验参数，便于后续分析和论文写作
"""
import os
from pathlib import Path

# ========== 基础路径配置 ==========
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR.parent / '01-Preprocessing'
DATA_FILE = DATA_DIR / 'data_preprocessed.csv'
RESULTS_DIR = BASE_DIR / 'results'
RESULTS_DIR.mkdir(exist_ok=True)
MODELS_DIR = BASE_DIR / 'models'
MODELS_DIR.mkdir(exist_ok=True)

# ========== 实验参数（基于plan.txt）==========
# I. AI模型（精简版）
MODELS = ['ARIMA', 'LSTM', '1D-CNN', 'Transformer']

# II. 序列长度（输入历史数据长度）- 精简版
# 时间步 = 5秒
SEQ_LENGTHS = [6, 60, 120, 360]
SEQ_LENGTHS_DESC = {
    6: '极短期（0.5分钟，捕捉瞬时模式）',
    60: '短期（5分钟，捕捉短期趋势）',
    120: '中期（10分钟，捕捉中期模式）',
    360: '长期（30分钟，捕捉长期趋势）'
}

# III. 预测长度（预测未来步数）- 精简版
PRED_LENGTHS = [1, 6, 120, 360]
PRED_LENGTHS_DESC = {
    1: '极短期（5秒，单步预测）',
    6: '短期（0.5分钟）',
    120: '中期（10分钟）',
    360: '长期（30分钟）'
}

# 时间步对应关系
TIME_STEP_SECONDS = 5  # 每个时间步 = 5秒

# ========== 数据划分 ==========
RANDOM_SEED = 42
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15
SHUFFLE = False  # 时间序列数据不打乱

# ========== 数据预处理配置 ==========
# 前N列为元数据（需要跳过）
METADATA_COLS = 5  # 前5列：管片号码, 管理行程, 记录日期, 记录时刻, 系统掘进状态

# ========== 特征名称映射（用于绘图和分析）==========
# 特征索引到名称的映射（从第0个特征开始，即跳过元数据后的第一个特征）
FEATURE_NAME_MAPPING = {
    0: ('贯入度', 'Penetration Rate', 'mm/min'),
    1: ('推进压力（上）', 'Thrust Pressure (Top)', 'MPa'),
    2: ('推进压力（右）', 'Thrust Pressure (Right)', 'MPa'),
    3: ('推进压力（下）', 'Thrust Pressure (Bottom)', 'MPa'),
    4: ('推进压力（左）', 'Thrust Pressure (Left)', 'MPa'),
    5: ('土舱土压（右上）', 'Chamber Pressure (Top-Right)', 'MPa'),
    6: ('土舱土压（右下）', 'Chamber Pressure (Bottom-Right)', 'MPa'),
    7: ('土舱土压（左上）', 'Chamber Pressure (Top-Left)', 'MPa'),
    8: ('土舱土压（左下）', 'Chamber Pressure (Bottom-Left)', 'MPa'),
    9: ('No.16推进千斤顶速度', 'Jack 16 Speed', 'mm/min'),
    10: ('No.4推进千斤顶速度', 'Jack 4 Speed', 'mm/min'),
    11: ('No.8推进千斤顶速度', 'Jack 8 Speed', 'mm/min'),
    12: ('No.12推进千斤顶速度', 'Jack 12 Speed', 'mm/min'),
    13: ('推进油缸总推力', 'Total Thrust Force', 'kN'),
    14: ('No.16推进千斤顶行程', 'Jack 16 Stroke', 'mm'),
    15: ('No.4推进千斤顶行程', 'Jack 4 Stroke', 'mm'),
    16: ('No.8推进千斤顶行程', 'Jack 8 Stroke', 'mm'),
    17: ('No.12推进千斤顶行程', 'Jack 12 Stroke', 'mm'),
    18: ('千斤顶行程差 上下', 'Jack Stroke Difference', 'mm'),
    19: ('推进平均速度', 'Average Thrust Speed', 'mm/min'),
    20: ('刀盘转速', 'Cutterhead Rotation Speed', 'r/min'),
    21: ('刀盘扭矩', 'Cutterhead Torque', 'kN·m'),
    22: ('No.1刀盘电机扭矩', 'Motor 1 Torque', '%'),
    23: ('No.2刀盘电机扭矩', 'Motor 2 Torque', '%'),
    24: ('No.3刀盘电机扭矩', 'Motor 3 Torque', '%'),
    25: ('No.4刀盘电机扭矩', 'Motor 4 Torque', '%'),
    26: ('No.5刀盘电机扭矩', 'Motor 5 Torque', '%'),
    27: ('No.6刀盘电机扭矩', 'Motor 6 Torque', '%'),
    28: ('No.7刀盘电机扭矩', 'Motor 7 Torque', '%'),
    29: ('No.8刀盘电机扭矩', 'Motor 8 Torque', '%'),
    30: ('No.9刀盘电机扭矩', 'Motor 9 Torque', '%'),
    31: ('No.10刀盘电机扭矩', 'Motor 10 Torque', '%'),
}

# 关键特征索引（用于绘图展示）
KEY_FEATURES = {
    '贯入度': 0,
    '推进压力': 1,  # 使用推进压力（上）作为代表
    '刀盘转速': 20
}

# ========== 模型超参数配置 ==========
MODEL_CONFIGS = {
    'LSTM': {
        'hidden_size': 64,
        'num_layers': 2,
        'learning_rate': 1e-3,
        'batch_size': 64,
        'epochs': 20,
        'dropout': 0.1
    },
    '1D-CNN': {
        'num_filters': 64,
        'kernel_size': 3,
        'num_layers': 2,
        'learning_rate': 1e-3,
        'batch_size': 64,
        'epochs': 20,
        'dropout': 0.1
    },
    'Transformer': {
        'd_model': 64,
        'nhead': 4,
        'num_layers': 2,
        'dim_feedforward': 256,
        'learning_rate': 1e-3,
        'batch_size': 64,
        'epochs': 20,
        'dropout': 0.1
    },
    'ARIMA': {
        'order': (2, 1, 2),  # (p, d, q)
        'seasonal_order': None
    }
}

# ========== 评估指标配置（基于plan.txt）==========
# IV. 评估指标
METRICS = ['MSE', 'MAE', 'RMSE', 'MAPE', 'R2']  # 精度类指标

# 性能类指标（自动记录）
PERFORMANCE_METRICS = ['inference_time_ms', 'memory_usage_mb', 'model_size_params', 'model_size_mb']

# ========== 结果保存配置 ==========
# 结果文件命名格式：{model}_{seq_len}_{pred_len}_results.pkl
# 汇总文件：experiment_summary.csv

# ========== 辅助函数 ==========
def get_feature_name(feature_idx, lang='cn'):
    """
    获取特征名称
    
    Args:
        feature_idx: 特征索引（从0开始，已跳过元数据）
        lang: 语言，'cn'为中文，'en'为英文
    
    Returns:
        str: 特征名称
    """
    if feature_idx in FEATURE_NAME_MAPPING:
        if lang == 'cn':
            return FEATURE_NAME_MAPPING[feature_idx][0]
        else:
            return FEATURE_NAME_MAPPING[feature_idx][1]
    return f'特征{feature_idx}'

def get_feature_unit(feature_idx):
    """
    获取特征单位
    
    Args:
        feature_idx: 特征索引
    
    Returns:
        str: 特征单位
    """
    if feature_idx in FEATURE_NAME_MAPPING:
        return FEATURE_NAME_MAPPING[feature_idx][2]
    return ''

def get_key_feature_index(feature_name):
    """
    获取关键特征的索引
    
    Args:
        feature_name: 特征名称（中文）
    
    Returns:
        int: 特征索引，如果不存在返回None
    """
    return KEY_FEATURES.get(feature_name)

def get_total_experiments():
    """获取总实验数"""
    return len(MODELS) * len(SEQ_LENGTHS) * len(PRED_LENGTHS)

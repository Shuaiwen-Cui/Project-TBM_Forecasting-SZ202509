"""
ARIMA模型实现
基于plan.txt的配置：order=(2,1,2)
注意：ARIMA是单变量模型，需要对每个特征分别建模
"""
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from pathlib import Path
from .base_model import BaseModel
import warnings
warnings.filterwarnings('ignore')


class ARIMAModel(BaseModel):
    """ARIMA模型（对每个特征分别建模）"""
    
    def __init__(self):
        self.models = {}  # 每个特征一个ARIMA模型
        self.n_features = None
        self.seq_len = None
        self.pred_len = None
        self.order = (2, 1, 2)
    
    def train(self, X_train, Y_train, X_val=None, Y_val=None, **kwargs):
        """
        训练ARIMA模型
        
        Args:
            X_train: 训练输入 (n_samples, seq_len, n_features)
            Y_train: 训练输出 (n_samples, pred_len, n_features)
            X_val: 验证输入（可选，ARIMA不使用）
            Y_val: 验证输出（可选，ARIMA不使用）
            **kwargs: 超参数
                - order: ARIMA参数(p,d,q)，默认(2,1,2)
        """
        # 获取参数
        self.order = kwargs.get('order', (2, 1, 2))
        
        # 保存维度信息
        self.n_features = X_train.shape[-1]
        self.seq_len = X_train.shape[1]
        self.pred_len = Y_train.shape[1]
        
        # ARIMA需要对每个特征分别建模
        # 使用训练数据的最后一个时间步作为输入序列
        # 将X_train转换为时间序列格式
        train_sequences = X_train[:, -1, :]  # (n_samples, n_features) - 使用最后一个时间步
        
        # 对每个特征训练ARIMA模型
        self.models = {}
        
        for feat_idx in range(self.n_features):
            try:
                # 提取该特征的时间序列
                feature_series = train_sequences[:, feat_idx]
                
                # 训练ARIMA模型
                model = ARIMA(feature_series, order=self.order)
                fitted_model = model.fit()
                
                self.models[feat_idx] = fitted_model
            except Exception as e:
                # 如果某个特征训练失败，使用简单均值预测
                print(f"警告: 特征{feat_idx}的ARIMA模型训练失败: {e}")
                self.models[feat_idx] = None
        
        return {'loss': []}  # ARIMA不返回训练历史
    
    def predict(self, X, pred_len=None):
        """
        预测。为保证与深度学习模型公平对比，每个测试样本使用该样本的输入序列
        (seq_len) 通过 apply() 更新状态后再 forecast，使 ARIMA 与 LSTM/CNN/Transformer
        使用相同的输入信息。
        """
        if pred_len is None:
            pred_len = self.pred_len
        
        n_samples = X.shape[0]
        predictions = np.zeros((n_samples, pred_len, self.n_features))
        
        for sample_idx in range(n_samples):
            last_values = X[sample_idx, -1, :]
            for feat_idx in range(self.n_features):
                if self.models.get(feat_idx) is not None:
                    try:
                        # 使用该样本的输入序列 (长度 seq_len) 作为条件，与深度学习模型对齐
                        history = np.asarray(X[sample_idx, :, feat_idx], dtype=np.float64)
                        applied = self.models[feat_idx].apply(history)
                        forecast = applied.forecast(steps=pred_len)
                        predictions[sample_idx, :, feat_idx] = forecast
                    except Exception:
                        # 序列过短或 apply 失败时回退为从拟合模型直接 forecast（未使用该样本输入）
                        try:
                            predictions[sample_idx, :, feat_idx] = self.models[feat_idx].forecast(steps=pred_len)
                        except Exception:
                            predictions[sample_idx, :, feat_idx] = last_values[feat_idx]
                else:
                    predictions[sample_idx, :, feat_idx] = last_values[feat_idx]
        
        return predictions
    
    def save(self, filepath):
        """保存模型"""
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump({
                'models': self.models,
                'n_features': self.n_features,
                'seq_len': self.seq_len,
                'pred_len': self.pred_len,
                'order': self.order
            }, f)
    
    def load(self, filepath):
        """加载模型"""
        import pickle
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.models = data['models']
            self.n_features = data['n_features']
            self.seq_len = data['seq_len']
            self.pred_len = data['pred_len']
            self.order = data['order']

"""
模型基类 - 统一接口
所有模型都需要实现这个基类
"""
from abc import ABC, abstractmethod


class BaseModel(ABC):
    """模型基类"""
    
    @abstractmethod
    def train(self, X_train, Y_train, X_val=None, Y_val=None, **kwargs):
        """
        训练模型
        
        Args:
            X_train: 训练输入 (n_samples, seq_len, n_features)
            Y_train: 训练输出 (n_samples, pred_len, n_features)
            X_val: 验证输入（可选）
            Y_val: 验证输出（可选）
            **kwargs: 其他参数
        
        Returns:
            dict: 训练历史（可选）
        """
        pass
    
    @abstractmethod
    def predict(self, X, pred_len=None):
        """
        预测
        
        Args:
            X: 输入 (n_samples, seq_len, n_features)
            pred_len: 预测长度（如果模型需要）
        
        Returns:
            np.ndarray: 预测结果 (n_samples, pred_len, n_features)
        """
        pass
    
    @abstractmethod
    def save(self, filepath):
        """保存模型"""
        pass
    
    @abstractmethod
    def load(self, filepath):
        """加载模型"""
        pass

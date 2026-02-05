"""
1D-CNN模型实现
基于plan.txt的配置：num_filters=64, kernel_size=3, num_layers=2, dropout=0.1
"""
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from .base_model import BaseModel


class CNN1DModel(BaseModel):
    """1D-CNN模型"""
    
    def __init__(self):
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.n_features = None
        self.seq_len = None
        self.pred_len = None
    
    def train(self, X_train, Y_train, X_val=None, Y_val=None, **kwargs):
        """
        训练1D-CNN模型
        
        Args:
            X_train: 训练输入 (n_samples, seq_len, n_features)
            Y_train: 训练输出 (n_samples, pred_len, n_features)
            X_val: 验证输入（可选）
            Y_val: 验证输出（可选）
            **kwargs: 超参数
                - num_filters: 卷积核数量，默认64
                - kernel_size: 卷积核大小，默认3
                - num_layers: 层数，默认2
                - learning_rate: 学习率，默认1e-3
                - batch_size: 批次大小，默认64
                - epochs: 训练轮数，默认20
                - dropout: Dropout率，默认0.1
        """
        # 获取参数
        num_filters = kwargs.get('num_filters', 64)
        kernel_size = kwargs.get('kernel_size', 3)
        num_layers = kwargs.get('num_layers', 2)
        learning_rate = kwargs.get('learning_rate', 1e-3)
        batch_size = kwargs.get('batch_size', 64)
        epochs = kwargs.get('epochs', 20)
        dropout = kwargs.get('dropout', 0.1)
        
        # 保存维度信息
        self.n_features = X_train.shape[-1]
        self.seq_len = X_train.shape[1]
        self.pred_len = Y_train.shape[1]
        
        # 创建模型
        self.model = CNN1DNetwork(
            input_size=self.n_features,
            seq_len=self.seq_len,
            num_filters=num_filters,
            kernel_size=kernel_size,
            num_layers=num_layers,
            output_size=self.n_features,
            pred_len=self.pred_len,
            dropout=dropout
        ).to(self.device)
        
        # 优化器和损失函数
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        
        # 转换为PyTorch张量
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        Y_train_tensor = torch.FloatTensor(Y_train).to(self.device)
        
        # 训练
        self.model.train()
        history = {'loss': []}
        
        for epoch in range(epochs):
            total_loss = 0
            n_batches = 0
            
            for i in range(0, len(X_train_tensor), batch_size):
                batch_X = X_train_tensor[i:i+batch_size]
                batch_Y = Y_train_tensor[i:i+batch_size]
                
                optimizer.zero_grad()
                output = self.model(batch_X)
                loss = criterion(output, batch_Y)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                n_batches += 1
            
            avg_loss = total_loss / n_batches if n_batches > 0 else 0
            history['loss'].append(avg_loss)
        
        return history
    
    def predict(self, X, pred_len=None):
        """
        预测
        
        Args:
            X: 输入 (n_samples, seq_len, n_features)
            pred_len: 预测长度（如果模型需要，但CNN已经固定了pred_len）
        
        Returns:
            np.ndarray: 预测结果 (n_samples, pred_len, n_features)
        """
        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        with torch.no_grad():
            output = self.model(X_tensor)
        
        return output.cpu().numpy()
    
    def save(self, filepath):
        """保存模型"""
        if self.model is not None:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'n_features': self.n_features,
                'seq_len': self.seq_len,
                'pred_len': self.pred_len
            }, filepath)
    
    def load(self, filepath):
        """加载模型"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.n_features = checkpoint['n_features']
        self.seq_len = checkpoint['seq_len']
        self.pred_len = checkpoint['pred_len']


class CNN1DNetwork(nn.Module):
    """1D-CNN网络结构"""
    
    def __init__(self, input_size, seq_len, num_filters, kernel_size, num_layers, 
                 output_size, pred_len, dropout=0.1):
        super(CNN1DNetwork, self).__init__()
        self.pred_len = pred_len
        
        # 构建卷积层
        conv_layers = []
        in_channels = input_size
        
        for i in range(num_layers):
            conv_layers.append(
                nn.Conv1d(in_channels=in_channels, out_channels=num_filters, 
                        kernel_size=kernel_size, padding=kernel_size//2)
            )
            conv_layers.append(nn.ReLU())
            conv_layers.append(nn.Dropout(dropout))
            in_channels = num_filters
        
        self.conv_layers = nn.Sequential(*conv_layers)
        
        # 计算卷积后的序列长度（由于padding，长度不变）
        conv_output_len = seq_len
        
        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(num_filters * conv_output_len, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, output_size * pred_len)
        )
    
    def forward(self, x):
        # x: (batch, seq_len, input_size)
        x = x.transpose(1, 2)  # (batch, input_size, seq_len)
        
        # 卷积
        conv_out = self.conv_layers(x)  # (batch, num_filters, seq_len)
        
        # 展平
        flattened = conv_out.view(conv_out.size(0), -1)  # (batch, num_filters * seq_len)
        
        # 全连接
        output = self.fc(flattened)  # (batch, output_size * pred_len)
        
        # 重塑为 (batch, pred_len, output_size)
        output = output.view(output.size(0), self.pred_len, -1)
        
        return output

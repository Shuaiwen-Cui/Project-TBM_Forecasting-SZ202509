"""
Transformer模型实现
基于plan.txt的配置：d_model=64, nhead=4, num_layers=2, dim_feedforward=256, dropout=0.1
"""
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from .base_model import BaseModel


class TransformerModel(BaseModel):
    """Transformer模型"""
    
    def __init__(self):
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.n_features = None
        self.seq_len = None
        self.pred_len = None
    
    def train(self, X_train, Y_train, X_val=None, Y_val=None, **kwargs):
        """
        训练Transformer模型
        
        Args:
            X_train: 训练输入 (n_samples, seq_len, n_features)
            Y_train: 训练输出 (n_samples, pred_len, n_features)
            X_val: 验证输入（可选）
            Y_val: 验证输出（可选）
            **kwargs: 超参数
                - d_model: 模型维度，默认64
                - nhead: 注意力头数，默认4
                - num_layers: 编码器层数，默认2
                - dim_feedforward: 前馈网络维度，默认256
                - learning_rate: 学习率，默认1e-3
                - batch_size: 批次大小，默认64
                - epochs: 训练轮数，默认20
                - dropout: Dropout率，默认0.1
        """
        # 获取参数
        d_model = kwargs.get('d_model', 64)
        nhead = kwargs.get('nhead', 4)
        num_layers = kwargs.get('num_layers', 2)
        dim_feedforward = kwargs.get('dim_feedforward', 256)
        learning_rate = kwargs.get('learning_rate', 1e-3)
        batch_size = kwargs.get('batch_size', 64)
        epochs = kwargs.get('epochs', 20)
        dropout = kwargs.get('dropout', 0.1)
        seed = kwargs.get('random_seed', 42)
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        
        # 保存维度信息
        self.n_features = X_train.shape[-1]
        self.seq_len = X_train.shape[1]
        self.pred_len = Y_train.shape[1]
        
        # 创建模型
        self.model = TransformerNetwork(
            input_size=self.n_features,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
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
            pred_len: 预测长度（如果模型需要，但Transformer已经固定了pred_len）
        
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


class TransformerNetwork(nn.Module):
    """Transformer网络结构"""
    
    def __init__(self, input_size, d_model, nhead, num_layers, dim_feedforward,
                 output_size, pred_len, dropout=0.1):
        super(TransformerNetwork, self).__init__()
        self.pred_len = pred_len
        
        # 输入投影层
        self.input_projection = nn.Linear(input_size, d_model)
        
        # 位置编码（可学习的）
        self.pos_encoder = nn.Parameter(torch.randn(1000, d_model))  # 支持最大1000步
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 输出投影层
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, output_size * pred_len)
        )
    
    def forward(self, x):
        # x: (batch, seq_len, input_size)
        seq_len = x.size(1)
        
        # 输入投影
        x = self.input_projection(x)  # (batch, seq_len, d_model)
        
        # 位置编码
        pos_emb = self.pos_encoder[:seq_len, :].unsqueeze(0)  # (1, seq_len, d_model)
        x = x + pos_emb  # (batch, seq_len, d_model)
        
        # Transformer编码
        encoded = self.transformer_encoder(x)  # (batch, seq_len, d_model)
        
        # 使用最后一个时间步的输出
        last_output = encoded[:, -1, :]  # (batch, d_model)
        
        # 输出投影
        output = self.output_projection(last_output)  # (batch, output_size * pred_len)
        
        # 重塑为 (batch, pred_len, output_size)
        output = output.view(output.size(0), self.pred_len, -1)
        
        return output

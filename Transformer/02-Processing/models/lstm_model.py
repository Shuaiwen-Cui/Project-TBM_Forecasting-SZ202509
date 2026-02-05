"""
LSTM模型实现
基于plan.txt的配置：hidden_size=64, num_layers=2, dropout=0.1
"""
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from .base_model import BaseModel


class LSTMModel(BaseModel):
    """LSTM模型"""
    
    def __init__(self):
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.n_features = None
        self.seq_len = None
        self.pred_len = None
    
    def train(self, X_train, Y_train, X_val=None, Y_val=None, **kwargs):
        """
        训练LSTM模型
        
        Args:
            X_train: 训练输入 (n_samples, seq_len, n_features)
            Y_train: 训练输出 (n_samples, pred_len, n_features)
            X_val: 验证输入（可选）
            Y_val: 验证输出（可选）
            **kwargs: 超参数
                - hidden_size: 隐藏层维度，默认64
                - num_layers: 层数，默认2
                - learning_rate: 学习率，默认1e-3
                - batch_size: 批次大小，默认64
                - epochs: 训练轮数，默认20
                - dropout: Dropout率，默认0.1
        """
        # 获取参数
        hidden_size = kwargs.get('hidden_size', 64)
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
        self.model = LSTMNetwork(
            input_size=self.n_features,
            hidden_size=hidden_size,
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
            pred_len: 预测长度（如果模型需要，但LSTM已经固定了pred_len）
        
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
        
        # 需要重新创建模型结构（这里简化处理，实际应该保存模型结构）
        # 注意：实际使用时需要知道模型结构参数


class LSTMNetwork(nn.Module):
    """LSTM网络结构"""
    
    def __init__(self, input_size, hidden_size, num_layers, output_size, pred_len, dropout=0.1):
        super(LSTMNetwork, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.pred_len = pred_len
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # 全连接层：从hidden_size映射到output_size
        # 注意：在多变量预测中，input_size应该等于output_size
        self.fc = nn.Linear(hidden_size, output_size)
        
        # 保存维度信息，用于forward中的检查
        self.input_size = input_size
        self.output_size = output_size
    
    def forward(self, x):
        # x: (batch, seq_len, input_size)
        # 注意：在多变量预测中，input_size应该等于output_size
        assert self.input_size == self.output_size, \
            f"LSTM递归预测要求input_size({self.input_size}) == output_size({self.output_size})"
        
        lstm_out, (hidden, cell) = self.lstm(x)  # (batch, seq_len, hidden_size)
        
        # 使用最后一个时间步的输出和隐藏状态
        last_hidden = lstm_out[:, -1, :]  # (batch, hidden_size)
        
        # 预测pred_len步
        if self.pred_len == 1:
            # 单步预测，直接输出
            output = self.fc(last_hidden)  # (batch, output_size)
            return output.unsqueeze(1)  # (batch, 1, output_size)
        
        # 多步预测：递归预测
        outputs = []
        current_hidden = hidden  # (num_layers, batch, hidden_size)
        current_cell = cell  # (num_layers, batch, hidden_size)
        
        # 第一个预测
        first_output = self.fc(last_hidden)  # (batch, output_size)
        outputs.append(first_output)
        
        # 后续预测：使用前一个预测值作为输入
        for _ in range(self.pred_len - 1):
            # 将前一个输出作为输入（reshape为(batch, 1, input_size)）
            # 因为input_size == output_size，所以可以直接使用
            prev_output = outputs[-1].unsqueeze(1)  # (batch, 1, output_size) = (batch, 1, input_size)
            
            # 通过LSTM
            lstm_out_step, (current_hidden, current_cell) = self.lstm(
                prev_output, (current_hidden, current_cell)
            )
            
            # 全连接层
            step_output = self.fc(lstm_out_step[:, -1, :])  # (batch, output_size)
            outputs.append(step_output)
        
        return torch.stack(outputs, dim=1)  # (batch, pred_len, output_size)

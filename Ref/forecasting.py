import onnxruntime as ort
import numpy as np
import joblib
import os

# 加载模型和scaler
model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'AI-Model')
session = ort.InferenceSession(os.path.join(model_dir, 'model_transformer.onnx'))
scaler = joblib.load(os.path.join(model_dir, 'minmax_scaler.pkl'))

def predict(raw_input_data):
    """
    盾构机预测函数
    
    Args:
        raw_input_data: 原始输入数据，形状为 (1, 5, 31)
                       1个样本 × 5个时间步 × 31个特征
    
    Returns:
        output: 预测结果，形状为 (1, 31)，原始量纲
    """
    # 步骤1: 数据标准化
    input_data_scaled = scaler.transform(raw_input_data.reshape(-1, 31)).reshape(1, 5, 31).astype(np.float32)
    
    # 步骤2: 模型推理
    output_scaled = session.run(None, {session.get_inputs()[0].name: input_data_scaled})
    
    # 步骤3: 反标准化
    output = scaler.inverse_transform(output_scaled[0])
    
    return output

# 示例使用
if __name__ == "__main__":
    # 准备测试数据（模拟真实盾构机数据）
    raw_input_data = np.zeros((1, 5, 31))
    for i in range(31):
        min_val = scaler.data_min_[i]
        max_val = scaler.data_max_[i]
        raw_input_data[0, :, i] = np.random.uniform(min_val, max_val, 5)
    
    # 进行预测
    prediction = predict(raw_input_data)
    
    print(f"预测结果形状: {prediction.shape}")
    print(f"预测结果: {prediction.flatten()}")
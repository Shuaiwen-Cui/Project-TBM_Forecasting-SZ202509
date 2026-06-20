import onnxruntime as ort
import numpy as np
import joblib

# 加载 ONNX 模型
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
onnx_path = os.path.join(script_dir, 'model_transformer.onnx')
session = ort.InferenceSession(onnx_path)

# 加载训练时保存的scaler对象（假设已保存为minmax_scaler.pkl）
scaler_path = os.path.join(script_dir, 'minmax_scaler.pkl')
scaler = joblib.load(scaler_path)

# 准备原始输入数据（未归一化，形状需与训练一致）
raw_input_data = np.random.randn(1, 5, 31)  # 替换为你的实际原始数据

# 归一化输入数据
input_data_scaled = scaler.transform(raw_input_data.reshape(-1, 31)).reshape(1, 5, 31).astype(np.float32)

# ONNX 输入名
input_name = session.get_inputs()[0].name

# 推理
output_scaled = session.run(None, {input_name: input_data_scaled})

# 逆归一化输出结果
output = scaler.inverse_transform(output_scaled[0])

# 输出结果（原始量纲）
print("模型输出（原始量纲）:", output)

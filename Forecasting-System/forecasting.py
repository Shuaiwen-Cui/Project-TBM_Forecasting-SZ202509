import onnxruntime as ort
import numpy as np
import joblib
import os

# 加载模型和scaler
model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'AI-Model')
session = ort.InferenceSession(os.path.join(model_dir, 'model_transformer.onnx'))
scaler = joblib.load(os.path.join(model_dir, 'minmax_scaler.pkl'))

print("=" * 60)
print("机器学习推理流程演示")
print("=" * 60)

# 步骤1: 准备原始数据（模拟真实盾构机数据）
print("\n步骤1: 准备原始数据（原始量纲）")
print("   数据结构: 1个样本 × 5个时间步 × 31个特征")
# 按照每个特征各自的训练数据范围生成数据
raw_input_data = np.zeros((1, 5, 31))
for i in range(31):  # 遍历31个特征
    min_val = scaler.data_min_[i]
    max_val = scaler.data_max_[i]
    # 为每个特征在其自己的范围内生成5个时间步的数据
    raw_input_data[0, :, i] = np.random.uniform(min_val, max_val, 5)

print(f"   原始数据形状: {raw_input_data.shape}")
print(f"   原始数据范围: [{raw_input_data.min():.3f}, {raw_input_data.max():.3f}]")
print("   ✓ 这些是真实的盾构机参数值（如压力、速度、扭矩等）")

# 显示前几个特征的具体范围
print("\n   前5个特征的数据范围:")
for i in range(5):
    feature_data = raw_input_data[0, :, i]  # 5个时间步的数据
    print(f"     特征{i}: 时间步1={feature_data[0]:.3f}, 时间步2={feature_data[1]:.3f}, ..., 时间步5={feature_data[4]:.3f}")
    print(f"             范围: [{feature_data.min():.3f}, {feature_data.max():.3f}] (训练范围: [{scaler.data_min_[i]:.3f}, {scaler.data_max_[i]:.3f}])")

# 步骤2: 归一化（使用训练时的scaler）
print("\n步骤2: 归一化（使用训练时的scaler）")
input_data_scaled = scaler.transform(raw_input_data.reshape(-1, 31)).reshape(1, 5, 31).astype(np.float32)
print(f"   归一化后形状: {input_data_scaled.shape}")
print(f"   归一化后范围: [{input_data_scaled.min():.3f}, {input_data_scaled.max():.3f}]")
print("   ✓ 数据被压缩到[0,1]区间，符合模型训练时的数据分布")

# 步骤3: 模型推理
print("\n步骤3: 模型推理")
output_scaled = session.run(None, {session.get_inputs()[0].name: input_data_scaled})
print(f"   模型输出形状: {output_scaled[0].shape}")
print(f"   模型输出范围: [{output_scaled[0].min():.3f}, {output_scaled[0].max():.3f}]")
print("   ✓ 模型在归一化数据上进行推理，输出也是归一化的")

# 步骤4: 反归一化（还原到原始量纲）
print("\n步骤4: 反归一化（还原到原始量纲）")
output = scaler.inverse_transform(output_scaled[0])
print(f"   反归一化后形状: {output.shape}")
print(f"   反归一化后范围: [{output.min():.3f}, {output.max():.3f}]")
print("   ✓ 预测结果还原到原始量纲，具有实际物理意义")

print("\n" + "=" * 60)
print("推理流程总结:")
print("原始数据 → 归一化 → 模型推理 → 反归一化 → 最终结果")
print("=" * 60)

print(f"\n最终预测结果（原始量纲）:")
print(f"形状: {output.shape}")
print(f"数值: {output.flatten()}")

# 检查预测结果是否在训练范围内
print("\n" + "=" * 60)
print("预测结果范围检查:")
print("=" * 60)

out_of_range_count = 0
for i in range(31):
    pred_value = output[0, i]
    min_val = scaler.data_min_[i]
    max_val = scaler.data_max_[i]
    
    if pred_value < min_val or pred_value > max_val:
        out_of_range_count += 1
        status = "❌ 超出范围"
    else:
        status = "✅ 正常"
    
    print(f"特征{i:2d}: {pred_value:8.3f} (训练范围: [{min_val:6.3f}, {max_val:6.3f}]) {status}")

print(f"\n总结:")
print(f"  - 总特征数: 31")
print(f"  - 超出范围的特征数: {out_of_range_count}")
print(f"  - 正常范围的特征数: {31 - out_of_range_count}")
if out_of_range_count > 0:
    print(f"  - 警告: 有{out_of_range_count}个特征的预测值超出了训练时的数据范围")
    print(f"  - 这可能表明模型预测不够准确，或者输入数据与训练数据分布差异较大")
else:
    print(f"  - 所有预测值都在训练数据范围内，预测结果合理")
# TBM预测系统数据记录配置说明

## 配置参数

在`main.py`文件顶部，您可以通过以下参数控制数据记录功能：

```python
# 数据记录配置
ENABLE_DATA_LOGGING = True    # 是否启用数据记录功能
LOG_DATA_TO_FILE = True       # 是否记录数据到文件
LOG_PREDICTIONS_TO_FILE = True # 是否记录预测到文件
```

## 配置选项

### 1. 完全禁用记录
```python
ENABLE_DATA_LOGGING = False
```
- 不会创建任何记录文件
- 不会进行任何数据记录操作
- 性能最佳

### 2. 只记录数据
```python
ENABLE_DATA_LOGGING = True
LOG_DATA_TO_FILE = True
LOG_PREDICTIONS_TO_FILE = False
```
- 只记录拉取到的当前数据
- 不记录预测结果
- 生成文件：`tbm_data_YYYYMMDD_HHMMSS.txt`

### 3. 只记录预测
```python
ENABLE_DATA_LOGGING = True
LOG_DATA_TO_FILE = False
LOG_PREDICTIONS_TO_FILE = True
```
- 只记录预测结果
- 不记录当前数据
- 生成文件：`tbm_predictions_YYYYMMDD_HHMMSS.txt`

### 4. 记录所有（默认）
```python
ENABLE_DATA_LOGGING = True
LOG_DATA_TO_FILE = True
LOG_PREDICTIONS_TO_FILE = True
```
- 记录所有数据
- 生成两个文件：数据文件和预测文件

## 动态配置

运行时可以通过`set_logging_config()`方法动态修改配置：

```python
# 创建预测器实例
predictor = TBMPredictor()

# 禁用所有记录
predictor.set_logging_config(enable_logging=False)

# 只记录数据
predictor.set_logging_config(enable_logging=True, log_data=True, log_predictions=False)

# 只记录预测
predictor.set_logging_config(log_data=False, log_predictions=True)

# 记录所有
predictor.set_logging_config(log_data=True, log_predictions=True)
```

## 文件格式

### 数据文件格式
```
时间戳	步数	特征1_贯入度	特征2_推进区间的压力（上）	...	特征31_No.10刀盘电机扭矩
2025-09-23 19:37:25	1	0.642162	0.254790	...	0.167281
```

### 预测文件格式
```
时间戳	步数	预测1_贯入度	预测2_推进区间的压力（上）	...	预测31_No.10刀盘电机扭矩
2025-09-23 19:37:25	1	0.899616	0.141527	...	0.157251
```

## 文件命名规则

- 数据文件：`tbm_data_YYYYMMDD_HHMMSS.txt`
- 预测文件：`tbm_predictions_YYYYMMDD_HHMMSS.txt`
- 时间戳基于程序启动时间

## 性能考虑

- 记录功能会增加I/O开销
- 建议在测试和调试时启用，生产环境可选择性启用
- 文件会持续增长，注意磁盘空间管理

## 使用建议

1. **开发调试**：启用所有记录功能
2. **性能测试**：禁用记录功能
3. **数据分析**：只记录需要的数据类型
4. **生产环境**：根据需求选择性启用

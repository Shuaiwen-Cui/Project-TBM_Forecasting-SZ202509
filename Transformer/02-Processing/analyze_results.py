"""
结果分析脚本 - 基于plan.txt的评估指标进行分析
便于论文写作的数据分析
"""
import pandas as pd
import numpy as np
from pathlib import Path
import config

def analyze_results():
    """分析实验结果"""
    results_dir = config.RESULTS_DIR
    summary_file = results_dir / 'experiment_summary.csv'
    
    if not summary_file.exists():
        print("汇总表不存在，请先运行实验")
        return
    
    # 加载汇总表
    df = pd.read_csv(summary_file)
    print(f"\n总实验数: {len(df)}")
    print(f"\n汇总表预览:")
    print(df.head(10))
    
    # 按模型统计（基于plan.txt的评估指标）
    print("\n" + "="*60)
    print("按模型统计（平均值）- 基于plan.txt的评估指标")
    print("="*60)
    model_stats = df.groupby('model').agg({
        # 精度类指标
        'MSE': ['mean', 'std', 'min', 'max'],
        'MAE': ['mean', 'std', 'min', 'max'],
        'RMSE': ['mean', 'std', 'min', 'max'],
        'MAPE': ['mean', 'std', 'min', 'max'],
        'R2': ['mean', 'std', 'min', 'max'],
        # 性能类指标
        'inference_time_ms': ['mean', 'std', 'min', 'max'],
        'memory_usage_mb': ['mean', 'std', 'min', 'max'],
        'training_time_s': ['mean', 'std', 'min', 'max'],
        # 模型大小
        'model_size_params': ['mean', 'std', 'min', 'max'],
        'model_size_mb': ['mean', 'std', 'min', 'max']
    })
    print(model_stats)
    
    # 找出最佳模型（按R2）
    print("\n" + "="*60)
    print("最佳模型（按R²值，每个时间尺度组合）")
    print("="*60)
    best_by_r2 = df.loc[df.groupby(['seq_len', 'pred_len'])['R2'].idxmax()]
    print(best_by_r2[['model', 'seq_len', 'pred_len', 'R2', 'MSE', 'MAE', 'inference_time_ms']])
    
    # 找出最快模型（按推理时间）
    print("\n" + "="*60)
    print("最快模型（按推理时间，每个时间尺度组合）")
    print("="*60)
    fastest = df.loc[df.groupby(['seq_len', 'pred_len'])['inference_time_ms'].idxmin()]
    print(fastest[['model', 'seq_len', 'pred_len', 'inference_time_ms', 'R2', 'MSE']])
    
    # 按序列长度分析（基于plan.txt的时间尺度）
    print("\n" + "="*60)
    print("序列长度对性能的影响（平均R²值）")
    print("="*60)
    seq_len_impact = df.groupby(['model', 'seq_len'])['R2'].mean().unstack(level=0)
    print(seq_len_impact)
    
    # 按预测长度分析
    print("\n" + "="*60)
    print("预测长度对性能的影响（平均R²值）")
    print("="*60)
    pred_len_impact = df.groupby(['model', 'pred_len'])['R2'].mean().unstack(level=0)
    print(pred_len_impact)
    
    # 精度-效率权衡分析
    print("\n" + "="*60)
    print("精度-效率权衡（R² vs 推理时间）")
    print("="*60)
    for model in config.MODELS:
        model_data = df[df['model'] == model]
        avg_r2 = model_data['R2'].mean()
        avg_time = model_data['inference_time_ms'].mean()
        print(f"{model}: R²={avg_r2:.4f}, 推理时间={avg_time:.2f}ms")
    
    # 保存详细分析结果
    analysis_file = results_dir / 'analysis_summary.txt'
    with open(analysis_file, 'w', encoding='utf-8') as f:
        f.write("实验结果详细分析 - 基于plan.txt的评估指标\n")
        f.write("="*60 + "\n\n")
        f.write(f"总实验数: {len(df)}\n")
        f.write(f"模型数量: {len(df['model'].unique())}\n")
        f.write(f"序列长度组合: {len(df['seq_len'].unique())}\n")
        f.write(f"预测长度组合: {len(df['pred_len'].unique())}\n\n")
        
        f.write("="*60 + "\n")
        f.write("按模型统计（平均值）\n")
        f.write("="*60 + "\n")
        f.write(str(model_stats) + "\n\n")
        
        f.write("="*60 + "\n")
        f.write("最佳模型（按R²值）\n")
        f.write("="*60 + "\n")
        f.write(str(best_by_r2) + "\n\n")
        
        f.write("="*60 + "\n")
        f.write("最快模型（按推理时间）\n")
        f.write("="*60 + "\n")
        f.write(str(fastest) + "\n\n")
        
        f.write("="*60 + "\n")
        f.write("序列长度对性能的影响\n")
        f.write("="*60 + "\n")
        f.write(str(seq_len_impact) + "\n\n")
        
        f.write("="*60 + "\n")
        f.write("预测长度对性能的影响\n")
        f.write("="*60 + "\n")
        f.write(str(pred_len_impact) + "\n\n")
    
    print(f"\n详细分析结果已保存到: {analysis_file}")
    
    # 生成CSV格式的统计表（便于导入Excel和论文写作）
    stats_csv = results_dir / 'model_statistics.csv'
    model_stats.to_csv(stats_csv, encoding='utf-8-sig')
    print(f"模型统计表已保存到: {stats_csv}")

if __name__ == "__main__":
    analyze_results()

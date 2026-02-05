"""
主运行脚本 - 运行所有实验
基于plan.txt的实验设计，依次完成64个实验，记录所有关键数据
"""
import sys
from pathlib import Path

# 添加当前目录到路径
sys.path.append(str(Path(__file__).parent))

from experiment_runner import ExperimentRunner
import config

if __name__ == "__main__":
    print("="*60)
    print("实验框架启动 - 基于plan.txt的实验设计")
    print("="*60)
    print(f"实验配置:")
    print(f"  模型: {config.MODELS}")
    print(f"  序列长度: {config.SEQ_LENGTHS}")
    print(f"  预测长度: {config.PRED_LENGTHS}")
    print(f"  总实验数: {config.get_total_experiments()}")
    print("="*60)
    
    runner = ExperimentRunner()
    
    # 显示当前状态
    summary = runner.get_status_summary()
    print(f"\n当前状态:")
    print(f"  已完成: {summary['completed']}/{summary['total']}")
    print(f"  失败: {summary['failed']}")
    print(f"  待执行: {summary['pending']}")
    print()
    
    # 运行所有实验
    runner.run_all_experiments(skip_existing=True)
    
    # 显示最终状态
    summary = runner.get_status_summary()
    print(f"\n最终状态:")
    print(f"  已完成: {summary['completed']}/{summary['total']}")
    print(f"  失败: {summary['failed']}")
    print(f"  待执行: {summary['pending']}")
    print(f"\n结果保存在: {runner.results_dir}")
    print(f"汇总表: {runner.results_dir / 'experiment_summary.csv'}")
    print(f"状态文件: {runner.status_file}")

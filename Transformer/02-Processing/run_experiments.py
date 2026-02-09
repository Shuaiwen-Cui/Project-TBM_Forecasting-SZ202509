"""
主运行脚本 - 运行所有实验
基于plan.txt的实验设计，依次完成64个实验，记录所有关键数据。
通过命令行参数选择：全部执行（覆盖）或仅运行未完成的。
"""
import sys
import argparse
from pathlib import Path

# 添加当前目录到路径
sys.path.append(str(Path(__file__).parent))

# 无缓冲输出，便于实时看到进度（与 run_arima_only 一致）
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)

from experiment_runner import ExperimentRunner
import config


def _parse_args():
    parser = argparse.ArgumentParser(
        description="运行全部实验（64 个）。可选：全部覆盖或仅运行未完成的。"
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--overwrite", "-f",
        action="store_true",
        help="全部执行并覆盖已有结果（便于论文分析）"
    )
    group.add_argument(
        "--resume", "-r",
        action="store_true",
        help="仅运行未完成的实验，跳过已有结果（默认行为）"
    )
    args = parser.parse_args()
    # 默认：仅运行未完成的（resume）。加 -f/--overwrite 时覆盖。
    skip_existing = not args.overwrite
    return skip_existing


def _get_device_str():
    """返回当前 PyTorch 设备描述，便于确认是否使用 CUDA"""
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0) if torch.cuda.device_count() else "CUDA"
            return f"cuda:0 ({name})"
        return "cpu"
    except Exception:
        return "unknown"


def _out(msg, flush=True):
    print(msg, flush=flush)


if __name__ == "__main__":
    skip_existing = _parse_args()

    _out("=" * 60)
    _out("实验框架启动 - 基于plan.txt的实验设计")
    _out("=" * 60)
    _out("实验配置:")
    _out(f"  模型: {config.MODELS}")
    _out(f"  序列长度: {config.SEQ_LENGTHS}")
    _out(f"  预测长度: {config.PRED_LENGTHS}")
    _out(f"  总实验数: {config.get_total_experiments()}")
    _out(f"  模式: {'全部执行（覆盖已有）' if not skip_existing else '仅运行未完成的'}")
    _out(f"  PyTorch 设备: {_get_device_str()} (LSTM/1D-CNN/Transformer 将使用该设备)")
    _out("=" * 60)

    runner = ExperimentRunner()

    summary = runner.get_status_summary()
    _out("\n当前状态:")
    _out(f"  已完成: {summary['completed']}/{summary['total']}")
    _out(f"  失败: {summary['failed']}")
    _out(f"  待执行: {summary['pending']}")
    _out("")

    runner.run_all_experiments(skip_existing=skip_existing)

    summary = runner.get_status_summary()
    _out("\n最终状态:")
    _out(f"  已完成: {summary['completed']}/{summary['total']}")
    _out(f"  失败: {summary['failed']}")
    _out(f"  待执行: {summary['pending']}")
    _out(f"\n结果保存在: {runner.results_dir}")
    _out(f"汇总表: {runner.results_dir / 'experiment_summary.csv'}")
    _out(f"状态文件: {runner.status_file}")

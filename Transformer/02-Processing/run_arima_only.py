"""
仅重跑 ARIMA 实验脚本
用于在修复 ARIMA 预测逻辑后，重新运行 16 个 ARIMA 实验并覆盖原有结果。
新结果会覆盖 results/ 下对应的 *_results.pkl 以及 experiment_summary.csv 中的 ARIMA 行。
"""
import sys
import gc
import time
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, str(Path(__file__).parent))

import config
from experiment_runner import ExperimentRunner

def _out(msg, flush=True):
    print(msg, flush=flush)

if __name__ == "__main__":
    total = len(config.SEQ_LENGTHS) * len(config.PRED_LENGTHS)
    _out("=" * 60)
    _out("仅重跑 ARIMA 实验（覆盖原结果）")
    _out("=" * 60)
    _out(f"序列长度: {config.SEQ_LENGTHS}")
    _out(f"预测长度: {config.PRED_LENGTHS}")
    _out(f"ARIMA 实验数: {total}")
    _out("=" * 60)

    runner = ExperimentRunner()
    completed = 0
    failed = 0
    current = 0

    for seq_len in config.SEQ_LENGTHS:
        for pred_len in config.PRED_LENGTHS:
            current += 1
            _out(f"\n[{current}/{total}] 正在运行 ARIMA seq_len={seq_len} pred_len={pred_len} ...")
            t0 = time.perf_counter()
            result = runner.run_single_experiment(
                "ARIMA", seq_len, pred_len, skip_existing=False
            )
            elapsed = time.perf_counter() - t0
            if result is not None:
                completed += 1
                _out(f"    完成，耗时 {elapsed:.1f}s，R²={result['metrics']['R2']:.4f}")
            else:
                task_key = runner._get_task_key("ARIMA", seq_len, pred_len)
                task = runner.status["tasks"].get(task_key, {})
                if task.get("status") == "failed":
                    failed += 1
                    _out(f"    失败，耗时 {elapsed:.1f}s: {task.get('error', '')}")
                else:
                    _out(f"    跳过或异常，耗时 {elapsed:.1f}s")
            gc.collect()

    _out("")
    _out("=" * 60)
    _out(f"ARIMA 重跑完成: {completed}/{total} 成功, {failed} 失败")
    _out(f"结果目录: {runner.results_dir}")
    _out(f"汇总表: {runner.results_dir / 'experiment_summary.csv'}")
    _out("=" * 60)

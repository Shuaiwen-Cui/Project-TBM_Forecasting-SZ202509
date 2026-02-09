"""
实验运行器 - 主控制器
基于plan.txt的实验设计，依次完成64个实验，记录所有关键数据
"""
import sys
from pathlib import Path
import time
import gc
import json
from datetime import datetime
import numpy as np

# 添加当前目录到路径
sys.path.append(str(Path(__file__).parent))

import config
import data_loader
import utils


class ExperimentRunner:
    """实验运行器"""
    
    def __init__(self, status_file=None):
        """
        初始化实验运行器
        
        Args:
            status_file: 任务状态文件路径，默认在results目录下
        """
        self.results_dir = config.RESULTS_DIR
        self.results_dir.mkdir(exist_ok=True)
        self.status_file = status_file or (self.results_dir / 'experiment_status.json')
        self.status = self._load_status()
    
    def _load_status(self):
        """加载任务状态"""
        if self.status_file.exists():
            try:
                with open(self.status_file, 'r', encoding='utf-8') as f:
                    status = json.load(f)
                    # 将之前运行中的任务重置为pending（可能是程序异常退出）
                    for task_key, task in status.get('tasks', {}).items():
                        if task.get('status') == 'running':
                            task['status'] = 'pending'
                    return status
            except:
                pass
        return {
            'tasks': {},
            'last_update': None,
            'total': 0,
            'completed': 0,
            'failed': 0,
            'pending': 0
        }
    
    def _save_status(self):
        """保存任务状态"""
        self.status['last_update'] = datetime.now().isoformat()
        try:
            with open(self.status_file, 'w', encoding='utf-8') as f:
                json.dump(self.status, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"警告: 无法保存状态文件: {e}")
    
    def _get_task_key(self, model_name, seq_len, pred_len):
        """获取任务键"""
        return f"{model_name}_{seq_len}_{pred_len}"
    
    def _update_task_status(self, model_name, seq_len, pred_len, status, error=None):
        """更新任务状态"""
        task_key = self._get_task_key(model_name, seq_len, pred_len)
        
        if task_key not in self.status['tasks']:
            self.status['tasks'][task_key] = {
                'model': model_name,
                'seq_len': seq_len,
                'pred_len': pred_len,
                'status': 'pending',
                'created_at': datetime.now().isoformat()
            }
        
        task = self.status['tasks'][task_key]
        task['status'] = status
        task['updated_at'] = datetime.now().isoformat()
        
        if error:
            task['error'] = str(error)
        
        self._save_status()
    
    def _log(self, message, level='INFO'):
        """带时间戳的日志（立即刷新以便看到进度）"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"[{timestamp}] [{level}] {message}", flush=True)
    
    def _get_model_wrapper(self, model_name):
        """
        获取模型包装器
        
        注意：这里需要根据实际的模型实现来导入
        如果模型还未实现，可以返回None，实验会跳过
        """
        try:
            if model_name == 'ARIMA':
                from models.arima_model import ARIMAModel
                return ARIMAModel()
            elif model_name == 'LSTM':
                from models.lstm_model import LSTMModel
                return LSTMModel()
            elif model_name == '1D-CNN':
                from models.cnn1d_model import CNN1DModel
                return CNN1DModel()
            elif model_name == 'Transformer':
                from models.transformer_model import TransformerModel
                return TransformerModel()
            else:
                return None
        except ImportError as e:
            self._log(f"模型 {model_name} 未实现: {e}", 'WARNING')
            return None
    
    def run_single_experiment(self, model_name, seq_len, pred_len, skip_existing=True):
        """
        运行单个实验
        
        Args:
            model_name: 模型名称
            seq_len: 序列长度
            pred_len: 预测长度
            skip_existing: 是否跳过已完成的实验
        
        Returns:
            dict: 结果字典，如果失败返回None
        """
        task_key = self._get_task_key(model_name, seq_len, pred_len)
        
        # 检查是否已存在
        if skip_existing:
            result_file = self.results_dir / f"{model_name}_{seq_len}_{pred_len}_results.pkl"
            if result_file.exists():
                self._log(f"跳过已完成的实验: {task_key}")
                return None
        
        # 更新状态为运行中
        self._update_task_status(model_name, seq_len, pred_len, 'running')
        self._log(f"开始实验: {task_key}")
        
        try:
            # 加载数据（固定种子保证可复现）
            self._log(f"加载数据: seq_len={seq_len}, pred_len={pred_len}")
            data_dict = data_loader.prepare_data_for_experiment(
                config.DATA_FILE, seq_len, pred_len,
                random_seed=config.RANDOM_SEED
            )
            
            # 获取模型包装器
            model_wrapper = self._get_model_wrapper(model_name)
            if model_wrapper is None:
                raise ValueError(f"模型 {model_name} 未实现或导入失败")
            
            # 获取模型配置（注入随机种子以保证可复现）
            model_config = dict(config.MODEL_CONFIGS.get(model_name, {}))
            model_config.setdefault('random_seed', config.RANDOM_SEED)
            
            # 记录初始内存
            initial_memory = utils.get_memory_usage()
            
            # 训练模型
            self._log(f"训练模型: {model_name}")
            training_start = time.time()
            
            train_history = model_wrapper.train(
                data_dict['X_train'], data_dict['Y_train'],
                data_dict['X_val'], data_dict['Y_val'],
                **model_config
            )
            
            training_time = time.time() - training_start
            self._log(f"训练完成，耗时: {training_time:.2f}秒")
            
            # 预测
            self._log("进行预测...")
            y_pred = model_wrapper.predict(data_dict['X_test'], pred_len=pred_len)
            y_true = data_dict['Y_test']
            
            # 计算指标（基于plan.txt的精度类指标）
            # 将3D转换为2D用于计算指标
            y_true_flat = y_true.reshape(-1, y_true.shape[-1])
            y_pred_flat = y_pred.reshape(-1, y_pred.shape[-1])
            
            metrics = utils.calculate_metrics(y_true_flat, y_pred_flat)
            self._log(f"评估指标: R²={metrics['R2']:.4f}, MSE={metrics['MSE']:.4f}")
            
            # 测量性能指标（基于plan.txt的性能类指标）
            self._log("测量性能指标...")
            
            # 推理时间
            sample_idx = 0
            x_sample = data_dict['X_test'][sample_idx:sample_idx+1]
            inference_time = utils.measure_inference_time(
                model_wrapper, model_wrapper, x_sample, pred_len
            )
            
            # 批量推理时间
            batch_inference_time = utils.measure_batch_inference_time(
                model_wrapper, model_wrapper, data_dict['X_test'][:100], pred_len
            )
            
            # 内存占用
            peak_memory = utils.get_memory_usage() - initial_memory
            
            # 模型大小（基于plan.txt的模型大小指标）
            # 尝试获取实际的PyTorch模型对象
            actual_model = model_wrapper
            if hasattr(model_wrapper, 'model'):
                actual_model = model_wrapper.model
            
            model_size_params = utils.count_model_parameters(actual_model)
            
            # 计算模型文件大小
            try:
                import torch
                if hasattr(actual_model, 'state_dict'):
                    temp_file = self.results_dir / 'temp_model.pth'
                    torch.save(actual_model.state_dict(), temp_file)
                    model_size_mb = temp_file.stat().st_size / 1024 / 1024
                    temp_file.unlink()
                else:
                    model_size_mb = model_size_params * 4 / 1024 / 1024
            except:
                model_size_mb = model_size_params * 4 / 1024 / 1024
            
            # 反归一化（用于可视化）
            try:
                scaler = data_dict['scaler']
                y_true_inv = scaler.inverse_transform(y_true_flat)
                y_pred_inv = scaler.inverse_transform(y_pred_flat)
            except:
                y_true_inv = y_true_flat
                y_pred_inv = y_pred_flat
            
            # 构建结果字典（记录所有关键数据，便于论文写作）
            result_dict = {
                'model_name': model_name,
                'feature_names': data_dict['feature_names'],
                'n_features': data_dict['n_features'],
                # 精度类指标（基于plan.txt）
                'metrics': metrics,
                # 性能类指标和模型大小（基于plan.txt）
                'costs': {
                    'model_size_params': int(model_size_params),
                    'model_size_mb': float(model_size_mb),
                    'inference_time_ms': float(inference_time),
                    'batch_inference_time_ms': float(batch_inference_time),
                    'memory_usage_mb': float(peak_memory),
                    'training_time_s': float(training_time)
                },
                'config': {
                    'seq_len': seq_len,
                    'pred_len': pred_len,
                    'model_config': model_config,
                    'train_ratio': config.TRAIN_RATIO,
                    'val_ratio': config.VAL_RATIO,
                    'test_ratio': config.TEST_RATIO,
                    'random_seed': config.RANDOM_SEED
                },
                'y_true': y_true,  # 归一化后的真实值
                'y_pred': y_pred,  # 归一化后的预测值
                'y_true_inv': y_true_inv,  # 反归一化后的真实值（用于绘图）
                'y_pred_inv': y_pred_inv,  # 反归一化后的预测值（用于绘图）
                'scaler': data_dict['scaler'],
                # 特征映射信息（便于绘图脚本使用）
                'feature_mapping': {
                    'key_features': config.KEY_FEATURES.copy(),
                    'feature_name_mapping': {idx: config.get_feature_name(idx, lang='cn') 
                                            for idx in range(data_dict['n_features'])}
                }
            }
            
            # 保存结果
            pkl_file = utils.save_experiment_result(
                result_dict, self.results_dir, model_name, seq_len, pred_len
            )
            self._log(f"结果已保存: {pkl_file}")
            
            # 更新汇总表
            utils.update_experiment_summary(self.results_dir, result_dict)
            self._log("汇总表已更新")
            
            # 更新状态为完成
            self._update_task_status(model_name, seq_len, pred_len, 'completed')
            
            # 清理内存
            del data_dict, model_wrapper, y_pred, y_true
            gc.collect()
            
            return result_dict
            
        except Exception as e:
            self._log(f"实验失败: {task_key}, 错误: {e}", 'ERROR')
            self._update_task_status(model_name, seq_len, pred_len, 'failed', error=str(e))
            return None
    
    def run_all_experiments(self, skip_existing=True):
        """
        运行所有实验（基于plan.txt的64个实验组合）
        
        Args:
            skip_existing: 是否跳过已完成的实验
        """
        total = config.get_total_experiments()
        self._log(f"开始运行所有实验，共 {total} 个实验组合")
        
        completed = 0
        failed = 0
        current = 0
        
        for model_name in config.MODELS:
            for seq_len in config.SEQ_LENGTHS:
                for pred_len in config.PRED_LENGTHS:
                    current += 1
                    task_key = self._get_task_key(model_name, seq_len, pred_len)
                    self._log(f"[{current}/{total}] 正在运行 {task_key} ...")
                    t0 = time.perf_counter()
                    result = self.run_single_experiment(
                        model_name, seq_len, pred_len, skip_existing=skip_existing
                    )
                    elapsed = time.perf_counter() - t0
                    if result is not None:
                        completed += 1
                        self._log(f"  完成，耗时 {elapsed:.1f}s，R²={result['metrics']['R2']:.4f}")
                    else:
                        task = self.status['tasks'].get(task_key, {})
                        if task.get('status') == 'failed':
                            failed += 1
                            self._log(f"  失败，耗时 {elapsed:.1f}s: {task.get('error', '')}")
                        else:
                            self._log(f"  跳过（已存在），耗时 {elapsed:.1f}s")
                    gc.collect()
        
        self._log(f"所有实验完成！共 {total} 个，已完成: {completed}, 失败: {failed}")
    
    def get_status_summary(self):
        """获取状态摘要"""
        total = config.get_total_experiments()
        completed = sum(1 for t in self.status['tasks'].values() if t.get('status') == 'completed')
        failed = sum(1 for t in self.status['tasks'].values() if t.get('status') == 'failed')
        pending = total - completed - failed
        
        return {
            'total': total,
            'completed': completed,
            'failed': failed,
            'pending': pending
        }

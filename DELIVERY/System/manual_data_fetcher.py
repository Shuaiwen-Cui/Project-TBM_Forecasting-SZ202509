#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
手动数据拉取脚本
根据Supporting-Material/盾构数据查询接口.txt和tbm_feature_mapping_correctified.csv
准确拉取TBM数据，显示原始unix时间戳、转换后的时间和31个特征
"""

import sys
import os
import json
import time
import requests
from datetime import datetime
import numpy as np

# 添加当前目录到Python路径，以便导入api_client
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

try:
    from api_client import TBMAPIClient
except ImportError as e:
    print(f"❌ 无法导入api_client: {e}")
    print(f"📁 当前目录: {current_dir}")
    print(f"📋 目录内容: {os.listdir(current_dir)}")
    sys.exit(1)

class ManualDataFetcher:
    def __init__(self):
        """初始化手动数据拉取器"""
        self.api_client = TBMAPIClient()
        
        # 根据tbm_feature_mapping_correctified.csv的特征映射
        self.feature_mapping = {
            1: ("贯入度", "date119"),
            2: ("推进区间的压力（上）", "date15"),
            3: ("推进区间的压力（右）", "date16"),
            4: ("推进区间的压力（下）", "date17"),
            5: ("推进区间的压力（左）", "date18"),
            6: ("土舱土压（右）", "date29"),
            7: ("土舱土压（右下）", "date30"),
            8: ("土舱土压（左）", "date31"),
            9: ("土舱土压（左下）", "date32"),
            10: ("No.16推进千斤顶速度", "date7"),
            11: ("No.4推进千斤顶速度", "date8"),
            12: ("No.8推进千斤顶速度", "date9"),
            13: ("No.12推进千斤顶速度", "date10"),
            14: ("推进油缸总推力", "date11"),
            15: ("No.16推进千斤顶行程", "date3"),
            16: ("No.4推进千斤顶行程", "date4"),
            17: ("No.8推进千斤顶行程", "date5"),
            18: ("No.12推进千斤顶行程", "date6"),
            19: ("推进平均速度", "date77"),
            20: ("刀盘转速", "date75"),
            21: ("刀盘扭矩", "date76"),
            22: ("No.1刀盘电机扭矩", "date46"),
            23: ("No.2刀盘电机扭矩", "date47"),
            24: ("No.3刀盘电机扭矩", "date48"),
            25: ("No.4刀盘电机扭矩", "date49"),
            26: ("No.5刀盘电机扭矩", "date50"),
            27: ("No.6刀盘电机扭矩", "date51"),
            28: ("No.7刀盘电机扭矩", "date52"),
            29: ("No.8刀盘电机扭矩", "date53"),
            30: ("No.9刀盘电机扭矩", "date54"),
            31: ("No.10刀盘电机扭矩", "date55")
        }
        
        print("🔧 手动数据拉取器初始化完成")
        print("=" * 60)
    
    def fetch_and_display_data(self):
        """拉取并显示数据"""
        try:
            print(f"📡 开始拉取数据... {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("-" * 60)
            
            # 获取最新数据
            api_response = self.api_client.fetch_data_by_time_range(
                begin_time=None,
                end_time=None,
                limit=2  # 获取2条记录，选择最新的
            )
            
            if not api_response or not api_response.get('data'):
                print("❌ 未获取到数据")
                return False
            
            records = api_response['data']
            print(f"📊 获取到 {len(records)} 条记录")
            print("-" * 60)
            
            # 选择ID最大的记录（最新的）
            latest_record = max(records, key=lambda x: int(x.get('id', 0)))
            
            # 显示原始记录信息
            print("🔍 原始记录信息:")
            print(f"   ID: {latest_record.get('id', 'N/A')}")
            print(f"   创建时间: {latest_record.get('createTime', 'N/A')}")
            print(f"   更新时间: {latest_record.get('updateTime', 'N/A')}")
            print(f"   盾构机编号: {latest_record.get('tbmId', 'N/A')}")
            
            # 解析data字段
            data_str = latest_record.get('data', '{}')
            try:
                data_dict = json.loads(data_str)
                print(f"   数据字段数量: {len(data_dict)}")
            except json.JSONDecodeError as e:
                print(f"   ❌ 数据解析失败: {e}")
                return False
            
            # 显示时间戳信息
            print("\n⏰ 时间戳信息:")
            create_time = latest_record.get('createTime', '')
            if create_time:
                try:
                    # 解析时间字符串
                    dt = datetime.strptime(create_time, "%Y-%m-%d %H:%M:%S")
                    unix_timestamp = int(dt.timestamp())
                    print(f"   原始时间字符串: {create_time}")
                    print(f"   转换后时间: {dt.strftime('%Y-%m-%d %H:%M:%S')}")
                    print(f"   Unix时间戳: {unix_timestamp}")
                    print(f"   当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                    print(f"   时间差: {int((datetime.now() - dt).total_seconds())} 秒前")
                except Exception as e:
                    print(f"   ⚠️  时间解析错误: {e}")
            
            # 显示31个特征值
            print(f"\n📋 31个特征值:")
            print("-" * 60)
            
            valid_count = 0
            for i in range(1, 32):  # 1到31
                if i in self.feature_mapping:
                    feature_name, transmission_name = self.feature_mapping[i]
                    
                    # 从data_dict中获取值
                    if transmission_name in data_dict:
                        value_str = data_dict[transmission_name]
                        try:
                            # 提取数值部分（去掉单位和括号）
                            if '(' in value_str:
                                numeric_value = float(value_str.split('(')[0])
                            else:
                                numeric_value = float(value_str)
                            
                            print(f"   {i:2d}. {feature_name:20s}: {numeric_value:>10.6f} ({transmission_name})")
                            valid_count += 1
                        except ValueError:
                            print(f"   {i:2d}. {feature_name:20s}: {'解析失败':>10s} ({transmission_name}: {value_str})")
                    else:
                        print(f"   {i:2d}. {feature_name:20s}: {'缺失':>10s} ({transmission_name})")
                else:
                    print(f"   {i:2d}. {'特征' + str(i):20s}: {'未定义':>10s}")
            
            print("-" * 60)
            print(f"✅ 数据拉取完成！有效特征: {valid_count}/31")
            print(f"📅 拉取时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("=" * 60)
            
            return True
            
        except Exception as e:
            print(f"❌ 数据拉取失败: {e}")
            return False
    
    def run_interactive(self):
        """交互式运行"""
        print("🚀 手动数据拉取器启动")
        print("💡 输入命令:")
        print("   'fetch' 或 'f' - 拉取数据")
        print("   'quit' 或 'q' - 退出")
        print("   'help' 或 'h' - 显示帮助")
        print("=" * 60)
        
        while True:
            try:
                command = input("\n🔧 请输入命令: ").strip().lower()
                
                if command in ['quit', 'q', 'exit']:
                    print("👋 再见！")
                    break
                elif command in ['fetch', 'f']:
                    self.fetch_and_display_data()
                elif command in ['help', 'h']:
                    print("💡 可用命令:")
                    print("   'fetch' 或 'f' - 拉取并显示最新数据")
                    print("   'quit' 或 'q' - 退出程序")
                    print("   'help' 或 'h' - 显示此帮助信息")
                elif command == '':
                    continue
                else:
                    print("❌ 未知命令，输入 'help' 查看可用命令")
                    
            except KeyboardInterrupt:
                print("\n👋 用户中断，退出程序")
                break
            except Exception as e:
                print(f"❌ 发生错误: {e}")

def main():
    """主函数"""
    print("🔧 TBM手动数据拉取器")
    print("=" * 60)
    
    fetcher = ManualDataFetcher()
    
    # 检查命令行参数
    if len(sys.argv) > 1:
        if sys.argv[1] in ['--fetch', '-f']:
            # 直接拉取一次数据
            fetcher.fetch_and_display_data()
        elif sys.argv[1] in ['--help', '-h']:
            print("💡 使用方法:")
            print("   python manual_data_fetcher.py          # 交互式模式")
            print("   python manual_data_fetcher.py --fetch   # 直接拉取一次数据")
            print("   python manual_data_fetcher.py --help   # 显示帮助")
        else:
            print(f"❌ 未知参数: {sys.argv[1]}")
            print("💡 使用 --help 查看帮助")
    else:
        # 交互式模式
        fetcher.run_interactive()

if __name__ == "__main__":
    main()
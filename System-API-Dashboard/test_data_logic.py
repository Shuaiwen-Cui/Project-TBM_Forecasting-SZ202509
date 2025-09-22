#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试新的数据逻辑 - 验证数据流和颜色显示
"""

import requests
import time
import json

def test_data_scenarios():
    """测试不同的数据场景"""
    print("🧪 测试数据逻辑...")
    print("=" * 60)
    
    try:
        # 测试 simple_server.py
        print("📡 测试 simple_server.py...")
        response = requests.get('http://localhost:8000/api/tbm-data', timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            features = data.get('features', [])
            
            print(f"✅ 获取到 {len(features)} 个特征数据")
            
            # 分析数据类型
            valid_count = 0
            predicted_count = 0
            missing_count = 0
            
            print("\n📊 数据状态分析:")
            for i, feature in enumerate(features):
                if feature is None:
                    status = "缺失"
                    missing_count += 1
                elif isinstance(feature, (int, float)):
                    status = "有效"
                    valid_count += 1
                elif isinstance(feature, dict):
                    if feature.get('predicted'):
                        status = f"填充 ({feature.get('reason', 'unknown')})"
                        predicted_count += 1
                    else:
                        status = "有效"
                        valid_count += 1
                else:
                    status = "未知"
                    missing_count += 1
                
                print(f"  特征{i+1:2d}: {status}")
            
            print(f"\n📈 统计结果:")
            print(f"  - 有效数据: {valid_count} (绿色)")
            print(f"  - 填充数据: {predicted_count} (橙色)")
            print(f"  - 缺失数据: {missing_count} (橙色)")
            
            # 验证数据逻辑
            print(f"\n🔍 逻辑验证:")
            if predicted_count > valid_count:
                print("✅ 填充数据多于有效数据 - 符合API历史数据特点")
            else:
                print("⚠️  有效数据多于填充数据 - 可能不符合实际情况")
            
            if missing_count == 0:
                print("✅ 没有缺失数据 - 符合整批数据特点")
            else:
                print(f"⚠️  有 {missing_count} 个缺失数据 - 可能不符合整批数据特点")
            
            return True
        else:
            print(f"❌ 请求失败: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False

def test_multiple_requests():
    """测试多次请求，观察数据变化"""
    print("\n🔄 测试多次请求...")
    print("=" * 60)
    
    for i in range(3):
        print(f"\n第 {i+1} 次请求:")
        try:
            response = requests.get('http://localhost:8000/api/tbm-data', timeout=5)
            if response.status_code == 200:
                data = response.json()
                features = data.get('features', [])
                
                # 统计状态
                predicted_count = sum(1 for f in features if isinstance(f, dict) and f.get('predicted'))
                valid_count = sum(1 for f in features if isinstance(f, (int, float)))
                missing_count = sum(1 for f in features if f is None)
                
                print(f"  填充: {predicted_count}, 有效: {valid_count}, 缺失: {missing_count}")
            else:
                print(f"  请求失败: {response.status_code}")
        except Exception as e:
            print(f"  错误: {e}")
        
        time.sleep(2)

def main():
    """主测试函数"""
    print("🚀 开始测试新的数据逻辑...")
    print("确保 simple_server.py 正在运行 (python simple_server.py)")
    print("=" * 60)
    
    # 测试数据场景
    success = test_data_scenarios()
    
    if success:
        # 测试多次请求
        test_multiple_requests()
        
        print("\n" + "=" * 60)
        print("✅ 测试完成！")
        print("\n📋 预期结果:")
        print("1. 大部分当前值应该显示橙色（填充数据）")
        print("2. 少数当前值可能显示绿色（实时数据）")
        print("3. 预测值总是显示黄色")
        print("4. 数据应该是整批的（要么全部有，要么全部没有）")
    else:
        print("\n❌ 测试失败，请确保服务器正在运行")

if __name__ == "__main__":
    main()

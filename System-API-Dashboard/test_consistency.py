#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试 app.py 和 simple_server.py 的逻辑一致性
"""

import requests
import time
import json

def test_app_server():
    """测试 app.py 服务器"""
    print("🔍 测试 app.py 服务器...")
    
    try:
        # 测试获取数据
        response = requests.get('http://localhost:5000/api/tbm-data', timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ app.py 数据获取成功: {len(data.get('features', []))} 个特征")
            
            # 分析数据类型
            features = data.get('features', [])
            valid_count = 0
            predicted_count = 0
            missing_count = 0
            
            for i, feature in enumerate(features):
                if feature is None:
                    missing_count += 1
                elif isinstance(feature, (int, float)):
                    valid_count += 1
                elif isinstance(feature, dict) and feature.get('predicted'):
                    predicted_count += 1
                else:
                    valid_count += 1
            
            print(f"   - 有效数据: {valid_count}")
            print(f"   - 预测数据: {predicted_count}")
            print(f"   - 缺失数据: {missing_count}")
            
            return True
        else:
            print(f"❌ app.py 响应错误: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ app.py 连接失败: {e}")
        return False

def test_simple_server():
    """测试 simple_server.py 服务器"""
    print("🔍 测试 simple_server.py 服务器...")
    
    try:
        # 测试获取数据
        response = requests.get('http://localhost:8000/api/tbm-data', timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ simple_server.py 数据获取成功: {len(data.get('features', []))} 个特征")
            
            # 分析数据类型
            features = data.get('features', [])
            valid_count = 0
            predicted_count = 0
            missing_count = 0
            
            for i, feature in enumerate(features):
                if feature is None:
                    missing_count += 1
                elif isinstance(feature, (int, float)):
                    valid_count += 1
                elif isinstance(feature, dict) and feature.get('predicted'):
                    predicted_count += 1
                else:
                    valid_count += 1
            
            print(f"   - 有效数据: {valid_count}")
            print(f"   - 预测数据: {predicted_count}")
            print(f"   - 缺失数据: {missing_count}")
            
            return True
        else:
            print(f"❌ simple_server.py 响应错误: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ simple_server.py 连接失败: {e}")
        return False

def main():
    """主测试函数"""
    print("🚀 开始测试两个服务器的逻辑一致性...")
    print("=" * 60)
    
    # 测试 app.py
    app_success = test_app_server()
    print()
    
    # 测试 simple_server.py
    simple_success = test_simple_server()
    print()
    
    # 总结
    print("=" * 60)
    if app_success and simple_success:
        print("✅ 两个服务器都运行正常，逻辑一致")
    elif app_success:
        print("⚠️  只有 app.py 运行正常")
    elif simple_success:
        print("⚠️  只有 simple_server.py 运行正常")
    else:
        print("❌ 两个服务器都无法连接")
    
    print("\n📋 使用建议:")
    print("1. 如果两个服务器都正常，推荐使用 simple_server.py（更轻量）")
    print("2. 如果需要更复杂的API功能，使用 app.py")
    print("3. 确保只运行一个服务器，避免端口冲突")

if __name__ == "__main__":
    main()

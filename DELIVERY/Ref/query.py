#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
盾构机数据预测系统 - 数据接口测试
"""

import requests
import json
import pandas as pd
from datetime import datetime, timedelta
import time

class TBMDataFetcher:
    """盾构机数据获取器"""
    
    def __init__(self, access_token="0c46137497ffdf0369037ada468fe2d3"):
        self.base_url = "https://szqhx.sinoccdc.com/qhx/shield/data/list"
        self.access_token = access_token
        self.session = requests.Session()
        
    def fetch_data(self, tbm_id="THDG24493", begin_time=None, end_time=None, limit=None):
        """
        从接口获取盾构机数据
        
        Args:
            tbm_id: 盾构机编号
            begin_time: 开始时间 (格式: "2025-09-07 24:00:00")
            end_time: 结束时间 (格式: "2025-09-09 24:00:00")
            limit: 限制返回条数
            
        Returns:
            dict: API响应数据
        """
        # 如果没有指定时间，默认查询最近1天的数据
        if not begin_time:
            end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            begin_time = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S")
        
        # 构建请求参数
        params = {
            "access-token": self.access_token
        }
        
        # 构建请求体
        payload = {
            "tbmId": tbm_id,
            "beginTime": begin_time,
            "endTime": end_time
        }
        
        if limit:
            payload["limit"] = limit
            
        try:
            print(f"正在请求数据...")
            print(f"盾构机ID: {tbm_id}")
            print(f"时间范围: {begin_time} 到 {end_time}")
            if limit:
                print(f"限制条数: {limit}")
            
            # 发送POST请求
            response = self.session.post(
                self.base_url,
                params=params,
                json=payload,
                timeout=30
            )
            
            print(f"HTTP状态码: {response.status_code}")
            
            # 检查响应状态
            if response.status_code == 200:
                data = response.json()
                print(f"请求成功! 返回码: {data.get('code', 'N/A')}")
                print(f"消息: {data.get('msg', 'N/A')}")
                
                if 'data' in data and data['data']:
                    print(f"获取到 {len(data['data'])} 条数据记录")
                    return data
                else:
                    print("警告: 返回的数据为空")
                    return data
            else:
                print(f"请求失败! 状态码: {response.status_code}")
                print(f"响应内容: {response.text}")
                return None
                
        except requests.exceptions.RequestException as e:
            print(f"请求异常: {e}")
            return None
        except json.JSONDecodeError as e:
            print(f"JSON解析错误: {e}")
            print(f"原始响应: {response.text}")
            return None
        except Exception as e:
            print(f"未知错误: {e}")
            return None
    
    def parse_data(self, api_response):
        """
        解析API返回的数据
        
        Args:
            api_response: API响应数据
            
        Returns:
            pd.DataFrame: 解析后的数据框
        """
        if not api_response or 'data' not in api_response:
            print("没有可解析的数据")
            return None
            
        records = []
        
        for item in api_response['data']:
            try:
                # 解析JSON数据字段
                data_json = json.loads(item['data'])
                
                # 创建记录
                record = {
                    'id': item['id'],
                    'tbm_id': item['tbmId'],
                    'origin_time': item['originTime'],
                    'my_time': item['myTime'],
                    'create_time': item['createTime'],
                    'update_time': item['updateTime']
                }
                
                # 添加所有数据字段
                for key, value in data_json.items():
                    # 提取数值部分（去掉单位）
                    if isinstance(value, str) and '(' in value:
                        try:
                            numeric_value = float(value.split('(')[0])
                            record[key] = numeric_value
                        except ValueError:
                            record[key] = value
                    else:
                        record[key] = value
                
                records.append(record)
                
            except Exception as e:
                print(f"解析记录时出错: {e}")
                continue
        
        if records:
            df = pd.DataFrame(records)
            print(f"成功解析 {len(df)} 条记录")
            print(f"数据列数: {len(df.columns)}")
            return df
        else:
            print("没有成功解析任何记录")
            return None

def test_latest_data():
    """测试获取最新数据"""
    print("=" * 50)
    print("测试获取最新数据")
    print("=" * 50)
    
    # 创建数据获取器
    fetcher = TBMDataFetcher()
    
    # 测试1: 获取最新5条数据并分析时间间隔
    print("\n测试1: 获取最新5条数据并分析时间间隔")
    print("-" * 50)
    
    start_time = time.time()
    response = fetcher.fetch_data(limit=5)
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    if response and 'data' in response and response['data']:
        print(f"✓ 成功获取到 {len(response['data'])} 条最新数据!")
        print(f"  - 耗时: {elapsed_time:.3f}秒")
        
        # 分析时间间隔
        print(f"\n📊 数据时间分析:")
        print("-" * 30)
        
        records = response['data']
        for i, record in enumerate(records):
            create_time = record.get('createTime', 'N/A')
            origin_time = record.get('originTime', 'N/A')
            record_id = record.get('id', 'N/A')
            
            print(f"第{i+1}条数据:")
            print(f"  - ID: {record_id}")
            print(f"  - 创建时间: {create_time}")
            print(f"  - 原始时间戳: {origin_time}")
            
            # 转换时间戳为可读格式
            if origin_time != 'N/A':
                try:
                    origin_timestamp = int(origin_time) / 1000  # 转换为秒
                    origin_datetime = datetime.fromtimestamp(origin_timestamp)
                    print(f"  - 原始时间: {origin_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
                except:
                    print(f"  - 原始时间: 解析失败")
            print()
        
        # 计算时间间隔
        if len(records) > 1:
            print("⏱️  时间间隔分析:")
            print("-" * 20)
            
            # 按创建时间排序
            sorted_records = sorted(records, key=lambda x: x.get('createTime', ''))
            
            for i in range(len(sorted_records) - 1):
                current_time = sorted_records[i].get('createTime', '')
                next_time = sorted_records[i + 1].get('createTime', '')
                
                if current_time and next_time:
                    try:
                        current_dt = datetime.strptime(current_time, '%Y-%m-%d %H:%M:%S')
                        next_dt = datetime.strptime(next_time, '%Y-%m-%d %H:%M:%S')
                        time_diff = next_dt - current_dt
                        
                        print(f"第{i+1}条 → 第{i+2}条: {time_diff.total_seconds():.0f}秒")
                    except Exception as e:
                        print(f"第{i+1}条 → 第{i+2}条: 时间解析失败 - {e}")
            
            # 计算总时间跨度
            if len(sorted_records) >= 2:
                first_time = sorted_records[0].get('createTime', '')
                last_time = sorted_records[-1].get('createTime', '')
                
                if first_time and last_time:
                    try:
                        first_dt = datetime.strptime(first_time, '%Y-%m-%d %H:%M:%S')
                        last_dt = datetime.strptime(last_time, '%Y-%m-%d %H:%M:%S')
                        total_span = last_dt - first_dt
                        
                        print(f"\n📈 总体统计:")
                        print(f"  - 数据条数: {len(records)}")
                        print(f"  - 时间跨度: {total_span.total_seconds():.0f}秒")
                        print(f"  - 平均间隔: {total_span.total_seconds()/(len(records)-1):.1f}秒")
                    except Exception as e:
                        print(f"总体统计计算失败: {e}")
    else:
        print("✗ 未获取到最新数据")
    
    # 测试2: 使用最近7天的时间范围
    print("\n测试2: 获取最近7天的数据")
    print("-" * 40)
    
    from datetime import datetime, timedelta
    end_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    begin_time_str = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d %H:%M:%S")
    
    start_time = time.time()
    response = fetcher.fetch_data(
        begin_time=begin_time_str,
        end_time=end_time_str,
        limit=3
    )
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    if response and 'data' in response and response['data']:
        print(f"✓ 成功获取到 {len(response['data'])} 条最近7天的数据!")
        for i, record in enumerate(response['data']):
            print(f"  第{i+1}条:")
            print(f"    - ID: {record.get('id', 'N/A')}")
            print(f"    - 创建时间: {record.get('createTime', 'N/A')}")
            print(f"    - 原始时间: {record.get('originTime', 'N/A')}")
        print(f"  - 耗时: {elapsed_time:.3f}秒")
    else:
        print("✗ 最近7天内没有数据")
    
    # 测试3: 获取最近30天的数据
    print("\n测试3: 获取最近30天的数据")
    print("-" * 40)
    
    begin_time_str = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d %H:%M:%S")
    
    start_time = time.time()
    response = fetcher.fetch_data(
        begin_time=begin_time_str,
        end_time=end_time_str,
        limit=5
    )
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    if response and 'data' in response and response['data']:
        print(f"✓ 成功获取到 {len(response['data'])} 条最近30天的数据!")
        print("  数据时间范围:")
        for i, record in enumerate(response['data']):
            print(f"  第{i+1}条: {record.get('createTime', 'N/A')} (ID: {record.get('id', 'N/A')})")
        print(f"  - 耗时: {elapsed_time:.3f}秒")
    else:
        print("✗ 最近30天内没有数据")

def test_connection():
    """测试接口连接"""
    print("=" * 50)
    print("盾构机数据接口连接测试")
    print("=" * 50)
    
    # 创建数据获取器
    fetcher = TBMDataFetcher()
    
    # 测试: 测量拉取数据的时间
    print("\n测试: 测量拉取一条数据的时间")
    print("-" * 30)
    
    # 记录开始时间
    start_time = time.time()
    
    response = fetcher.fetch_data(
        begin_time="2025-09-07 24:00:00",
        end_time="2025-09-09 24:00:00",
        limit=1
    )
    
    # 记录结束时间
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    # 显示时间统计
    print(f"\n⏱️  时间统计:")
    print(f"  - 总耗时: {elapsed_time:.3f} 秒")
    print(f"  - 总耗时: {elapsed_time*1000:.1f} 毫秒")
    
    if response:
        print("\n✓ 接口连接成功!")
        print(f"返回码: {response.get('code', 'N/A')}")
        print(f"消息: {response.get('msg', 'N/A')}")
        
        if 'data' in response and response['data']:
            print(f"✓ 成功获取到 {len(response['data'])} 条数据")
            
            # 显示第一条数据的基本信息
            first_record = response['data'][0]
            print(f"\n第一条数据信息:")
            print(f"  - ID: {first_record.get('id', 'N/A')}")
            print(f"  - 盾构机ID: {first_record.get('tbmId', 'N/A')}")
            print(f"  - 创建时间: {first_record.get('createTime', 'N/A')}")
            
            # 解析数据字段
            try:
                data_json = json.loads(first_record.get('data', '{}'))
                print(f"  - 数据字段数量: {len(data_json)}")
                print(f"  - 前5个数据字段:")
                for i, (key, value) in enumerate(list(data_json.items())[:5]):
                    print(f"    {key}: {value}")
            except Exception as e:
                print(f"  - 数据解析错误: {e}")
        else:
            print("⚠ 返回数据为空")
    else:
        print("✗ 接口连接失败")

def performance_test():
    """性能测试 - 多次测试获取平均时间"""
    print("\n" + "=" * 50)
    print("性能测试 - 多次测试获取平均时间")
    print("=" * 50)
    
    fetcher = TBMDataFetcher()
    test_times = []
    success_count = 0
    
    # 进行5次测试
    for i in range(5):
        print(f"\n第 {i+1} 次测试:")
        print("-" * 20)
        
        start_time = time.time()
        response = fetcher.fetch_data(
            begin_time="2025-09-07 24:00:00",
            end_time="2025-09-09 24:00:00",
            limit=1
        )
        end_time = time.time()
        
        elapsed_time = end_time - start_time
        test_times.append(elapsed_time)
        
        if response and 'data' in response and response['data']:
            success_count += 1
            print(f"✓ 成功 - 耗时: {elapsed_time:.3f}秒 ({elapsed_time*1000:.1f}毫秒)")
        else:
            print(f"✗ 失败 - 耗时: {elapsed_time:.3f}秒 ({elapsed_time*1000:.1f}毫秒)")
    
    # 计算统计信息
    if test_times:
        avg_time = sum(test_times) / len(test_times)
        min_time = min(test_times)
        max_time = max(test_times)
        
        print(f"\n📊 性能统计:")
        print(f"  - 测试次数: {len(test_times)}")
        print(f"  - 成功次数: {success_count}")
        print(f"  - 成功率: {success_count/len(test_times)*100:.1f}%")
        print(f"  - 平均耗时: {avg_time:.3f}秒 ({avg_time*1000:.1f}毫秒)")
        print(f"  - 最快耗时: {min_time:.3f}秒 ({min_time*1000:.1f}毫秒)")
        print(f"  - 最慢耗时: {max_time:.3f}秒 ({max_time*1000:.1f}毫秒)")
        
        # 评估性能
        if avg_time < 1.0:
            print(f"  - 性能评价: 🚀 优秀 (< 1秒)")
        elif avg_time < 3.0:
            print(f"  - 性能评价: ✅ 良好 (1-3秒)")
        elif avg_time < 5.0:
            print(f"  - 性能评价: ⚠️  一般 (3-5秒)")
        else:
            print(f"  - 性能评价: ❌ 较慢 (> 5秒)")

if __name__ == "__main__":
    test_latest_data()
    test_connection()
    performance_test()

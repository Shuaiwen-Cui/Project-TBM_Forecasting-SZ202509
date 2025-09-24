#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查app.py中的特征名称是否与CSV文件一致
"""

import csv
import sys

def check_feature_names():
    """检查特征名称对应关系"""
    
    # 从CSV文件读取特征名称
    csv_names = []
    with open('tbm_feature_mapping_correctified.csv', 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            csv_names.append(row['特征名'])
    
    # 从app.py读取特征名称
    app_names = [
        '贯入度', '推进区间的压力（上）', '推进区间的压力（右）', '推进区间的压力（下）', '推进区间的压力（左）',
        '土舱土压（右）', '土舱土压（右下）', '土舱土压（左）', '土舱土压（左下）',
        'No.16推进千斤顶速度', 'No.4推进千斤顶速度', 'No.8推进千斤顶速度', 'No.12推进千斤顶速度',
        '推进油缸总推力', 'No.16推进千斤顶行程', 'No.4推进千斤顶行程', 'No.8推进千斤顶行程', 'No.12推进千斤顶行程',
        '推进平均速度', '刀盘转速', '刀盘扭矩',
        'No.1刀盘电机扭矩', 'No.2刀盘电机扭矩', 'No.3刀盘电机扭矩', 'No.4刀盘电机扭矩', 'No.5刀盘电机扭矩',
        'No.6刀盘电机扭矩', 'No.7刀盘电机扭矩', 'No.8刀盘电机扭矩', 'No.9刀盘电机扭矩', 'No.10刀盘电机扭矩'
    ]
    
    print("🔍 检查特征名称对应关系...")
    print("=" * 60)
    
    # 检查数量
    print(f"📊 CSV文件特征数量: {len(csv_names)}")
    print(f"📊 app.py特征数量: {len(app_names)}")
    
    if len(csv_names) != len(app_names):
        print("❌ 特征数量不匹配！")
        return False
    
    # 逐个对比
    all_match = True
    for i, (csv_name, app_name) in enumerate(zip(csv_names, app_names)):
        if csv_name != app_name:
            print(f"❌ 第{i+1}个特征不匹配:")
            print(f"   CSV: '{csv_name}'")
            print(f"   APP: '{app_name}'")
            all_match = False
        else:
            print(f"✅ 第{i+1}个特征匹配: {csv_name}")
    
    print("=" * 60)
    if all_match:
        print("🎉 所有特征名称完全匹配！")
    else:
        print("⚠️  发现不匹配的特征名称！")
    
    return all_match

if __name__ == "__main__":
    check_feature_names()

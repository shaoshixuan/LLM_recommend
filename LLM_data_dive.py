
input_json_path = 'D:/user_behaviors_fixed.json'  # ← 请替换为你的JSON文件路径
output_root_dir = 'F:/LR/dataset/output_dir'  # 输出目录

import json
from collections import defaultdict


def process_json_to_tsv(input_path, output_dir):
    # 读取原始JSON数据
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 初始化存储结构
    inter_data = defaultdict(list)  # 存储行为数据（四种偏差类型分开）
    item_tags = defaultdict(lambda: defaultdict(set))  # 存储每个artist在不同bias_type下的标签

    # 遍历所有用户数据
    for user_id, user_info in data.items():
        user_id = str(user_id)  # 确保用户ID是字符串类型

        # 处理四种偏差类型
        for bias_type in ['control', 'anchor', 'peak-end', 'both']:
            behaviors = user_info['behavior_data'].get(bias_type, [])

            for record in behaviors:
                # 提取并转换所有字段为字符串
                artist_id = str(record.get('artistID', ''))
                tag_value = str(record.get('tagValue', ''))
                timestamp = str(record.get('timestamp', ''))
                weight = str(record.get('weight', ''))

                # 写入inter.tsv数据
                inter_data[bias_type].append({
                    'user_id': user_id,
                    'artist_id': artist_id,
                    'tag': tag_value,
                    'timestamp': timestamp,
                    'bias_type': bias_type,
                    'weight': weight
                })

                # 收集分偏差类型的item标签（保留首次出现的tag）
                if tag_value:
                    item_tags[bias_type][artist_id].add(tag_value)

    # 写入分偏差类型的inter.tsv文件
    for bias_type, rows in inter_data.items():
        file_path = f"{output_dir}/{bias_type}.inter.tsv"
        with open(file_path, 'w', encoding='utf-8', newline='') as f:
            f.write('\t'.join([
                'user_id', 'artist_id', 'tag', 'timestamp',
                'bias_type', 'weight'
            ]) + '\n')
            for row in rows:
                # 直接使用预处理的字符串字段
                f.write('\t'.join([
                    row['user_id'], row['artist_id'], row['tag'],
                    row['timestamp'], row['bias_type'], row['weight']
                ]) + '\n')

    # 写入分偏差类型的item.tags.tsv文件
    for bias_type in ['control', 'anchor', 'peak-end', 'both']:
        item_file_path = f"{output_dir}/{bias_type}.item.tags.tsv"
        with open(item_file_path, 'w', encoding='utf-8', newline='') as f:
            f.write('\t'.join(['artist_id', 'tag']) + '\n')
            for artist_id, tags in item_tags[bias_type].items():
                # 取第一个出现的tag作为代表
                primary_tag = next(iter(tags))
                f.write(f"{artist_id}\t{primary_tag}\n")


# 使用示例
process_json_to_tsv(
    input_path='D:/user_behaviors_fixed.json',
    output_dir='F:/LR/dataset/output_dir'
)
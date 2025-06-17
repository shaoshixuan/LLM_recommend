# from sklearn.model_selection import train_test_split
# import os
# import json
# import pandas as pd
# from typing import Dict, Any
#
#
# def clean_and_transform(raw_json_path: str, output_path: str):
#     """
#     数据清洗与字段重构主函数
#     :param raw_json_path: 原始JSON文件路径
#     :param output_path: 处理后的TSV文件路径
#     """
#     # 定义字段映射关系
#     FIELD_MAPPING = {
#         "high_tagValue": "tag",
#         "high_timestamp": "timestamp",
#         "userID": "user_id",
#         "artistID": "artist_id"
#     }
#
#     processed_data = []
#
#     # 逐行读取原始JSON数据
#     with open(raw_json_path, 'r', encoding='utf-8') as f:
#         for line_num, raw_line in enumerate(f, 1):
#             try:
#                 # 解析单行JSON
#                 record = json.loads(raw_line.strip())
#
#                 # 移除冗余时间戳字段
#                 for field in ['initial_timestamp', 'end_timestamp','initial_tagValue','end_tagValue']:
#                     record.pop(field, None)
#
#                 # 字段重命名
#                 transformed = {FIELD_MAPPING.get(k, k): v for k, v in record.items()}
#
#                 # 添加bias_type字段
#                 transformed['bias_type'] = None
#
#                 # 确保weight字段存在
#                 transformed.setdefault('weight', 0)
#
#                 processed_data.append(transformed)
#
#             except json.JSONDecodeError as e:
#                 print(f"第{line_num}行JSON解析失败: {str(e)}")
#             except Exception as e:
#                 print(f"处理第{line_num}行时发生错误: {str(e)}")
#
#     # 创建DataFrame并验证字段
#     df = pd.DataFrame(processed_data)
#     required_columns = ['user_id', 'artist_id', 'tag', 'timestamp', 'bias_type', 'weight']
#
#     if not set(required_columns).issubset(df.columns):
#         missing = set(required_columns) - set(df.columns)
#         raise ValueError(f"缺失必要字段: {missing}")
#
#     # 重新排列字段顺序
#     df = df[required_columns]
#
#     # 数据类型验证
#     type_check = {
#         'user_id': 'object',
#         'artist_id': 'object',
#         'tag': 'object',
#         'timestamp': 'object',
#         'bias_type': 'object',
#         'weight': 'float64'
#     }
#
#     for col, dtype in type_check.items():
#         if df[col].dtype != dtype:
#             print(f"警告: {col} 字段类型异常，正在转换...")
#             df[col] = df[col].astype(dtype)
#
#     # 保存处理结果
#     print(df)
#     df.to_csv(output_path, sep='\t', index=False)
#     # print(f"处理完成，保存至: {output_path}")
#     print(f"总记录数: {len(df)}")
#     print(f"字段类型验证通过: {all(type_check[col] == df[col].dtype for col in required_columns)}")
#
#
# if __name__ == "__main__":
#     # 输入输出路径配置
#     RAW_INPUT = 'D:/final_user_bias_data.json'
#     CLEAN_OUTPUT = 'D:/cleaned_data.tsv'
#
#     # 执行清洗流程
#     clean_and_transform(RAW_INPUT, CLEAN_OUTPUT)
#
# # 配置参数
# RAW_JSON_PATH = 'D:/cleaned_data.tsv'  # 原始JSON数据路径
# SYNTHETIC_DIRS = {  # 合成数据目录映射
#     'control': 'F:/LR/dataset/control',
#     'anchor': 'F:/LR/dataset/anchor',
#     'peak': 'F:/LR/dataset/peak',
#     'both': 'F:/LR/dataset/both'
# }
# PROCESSED_BASE_DIR = 'F:/LR/dataset/processed'  # 处理后数据基目录
# TEST_USER_FILE = 'D:/cold_start_data.json'  # 冷启动测试用户文件
# RANDOM_SEED = 42  # 随机种子
#
# BIAS_TYPES = list(SYNTHETIC_DIRS.keys())  # 偏差类型列表
#
# def load_cold_start_users(file_path):
#     """正确加载冷启动用户ID"""
#     with open(file_path, 'r') as f:
#         # 假设JSON文件内容是用户ID列表
#         user_data = json.load(f)
#         # 提取用户ID并转换为字符串集合
#         return {str(user['userID']) for user in user_data}
#
# def split_and_merge(raw_clean, synthetic_dir, bias_type):
#     """划分并合并数据"""
#     # 随机打乱数据
#     shuffled = raw_clean.sample(frac=1, random_state=RANDOM_SEED)
#
#     # 划分训练/验证集
#     train_df, val_df = train_test_split(
#         shuffled,
#         test_size=0.2,
#         random_state=RANDOM_SEED
#     )
#
#     # 加载合成数据
#     syn_path = os.path.join(synthetic_dir, 'inter.tsv')
#     syn_df = pd.read_csv(syn_path, sep='\t')
#
#     # 合并数据（按user_id和artist_id去重）
#     merged_train = pd.concat([train_df, syn_df]).drop_duplicates(
#         subset=['user_id', 'artist_id']
#     )
#     merged_val = pd.concat([val_df, syn_df]).drop_duplicates(
#         subset=['user_id', 'artist_id']
#     )
#
#     return merged_train, merged_val
#
# def main():
#     # 创建输出目录
#     for bias in BIAS_TYPES:
#         os.makedirs(os.path.join(PROCESSED_BASE_DIR, bias, 'train'), exist_ok=True)
#         os.makedirs(os.path.join(PROCESSED_BASE_DIR, bias, 'val'), exist_ok=True)
#
#     # 加载冷启动用户
#     test_users = load_cold_start_users(TEST_USER_FILE)
#
#     # 处理原始数据，将冷启动用户从原始数据中删去
#     raw_data = []
#     with open(RAW_JSON_PATH, 'r') as f:
#         for line in f:
#             try:
#                 record = json.loads(line)
#                 print(record)
#                 if record['userID'] not in test_users:
#                     raw_data.append(record)
#             except json.JSONDecodeError:
#                 continue
#
#     raw_clean = pd.DataFrame(raw_data)
#
#     # 处理每个偏差类型
#     for bias_type in BIAS_TYPES:
#         print(f"Processing {bias_type}...")
#
#         # 合成数据目录
#         synthetic_dir = SYNTHETIC_DIRS[bias_type]
#
#         # 划分合并数据
#         train_df, val_df = split_and_merge(
#             raw_clean.copy(),
#             synthetic_dir,
#             bias_type
#         )
#
#         # 保存结果
#         train_df.to_csv(
#             os.path.join(PROCESSED_BASE_DIR, bias_type, 'train', 'inter.tsv'),
#             sep='\t',
#             index=False
#         )
#         val_df.to_csv(
#             os.path.join(PROCESSED_BASE_DIR, bias_type, 'val', 'inter.tsv'),
#             sep='\t',
#             index=False
#         )
#
#
# if __name__ == '__main__':
#     main()

import os
import json
import pandas as pd
from sklearn.model_selection import train_test_split

# 全局配置参数
RAW_JSON_PATH = 'D:/final_user_bias_data.json'  # 原始JSON数据路径
CLEAN_OUTPUT = 'D:/cleaned_data.tsv'  # 清洗后输出路径
SYNTHETIC_DIRS = {  # 合成数据目录映射
    'control': 'F:/LR/dataset/control',
    'anchor': 'F:/LR/dataset/anchor',
    'peak': 'F:/LR/dataset/peak',
    'both': 'F:/LR/dataset/both'
}
PROCESSED_BASE_DIR = 'F:/LR/dataset/processed'  # 处理后数据基目录
TEST_USER_FILE = 'D:/cold_start_data.json'  # 冷启动测试用户文件
RANDOM_SEED = 42  # 随机种子
BIAS_TYPES = list(SYNTHETIC_DIRS.keys())  # 偏差类型列表


def load_cold_start_users(file_path: str) -> set:
    """加载并验证冷启动用户ID集合"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            user_data = json.load(f)
            # 验证数据结构
            if not isinstance(user_data, list):
                raise ValueError("冷启动用户文件应为列表格式")
            return {str(record.get('userID', '')) for record in user_data}
    except Exception as e:
        print(f"加载冷启动用户失败: {str(e)}")
        return set()


def clean_and_transform(raw_json_path: str, output_path: str, test_users: set):
    """数据清洗与字段重构（先过滤后转换）"""
    FIELD_MAPPING = {
        "high_tagValue": "tag",
        "high_timestamp": "timestamp",
        "userID": "user_id",
        "artistID": "artist_id"
    }

    processed_data = []

    with open(raw_json_path, 'r', encoding='utf-8') as f:
        for line_num, raw_line in enumerate(f, 1):
            try:
                record = json.loads(raw_line.strip())

                # 冷启动用户过滤（使用原始字段名）
                if record.get('userID') in test_users:
                    continue

                # 字段重命名（转换后字段名）
                transformed = {
                    FIELD_MAPPING[k]: v
                    for k, v in record.items()
                    if k in FIELD_MAPPING
                }

                # 补充新字段
                transformed['bias_type'] = None
                if 'weight' in record:
                    transformed['weight'] = record['weight']

                processed_data.append(transformed)

            except json.JSONDecodeError as e:
                print(f"第{line_num}行JSON解析失败: {str(e)}")
            except Exception as e:
                print(f"处理第{line_num}行错误: {str(e)}")

    # 数据验证
    if not processed_data:
        raise ValueError("清洗后数据为空，请检查输入文件和过滤条件")

    df = pd.DataFrame(processed_data)
    required_columns = ['user_id', 'artist_id', 'tag', 'timestamp', 'bias_type', 'weight']

    if not set(required_columns).issubset(df.columns):
        missing = set(required_columns) - set(df.columns)
        raise ValueError(f"缺失必要字段: {missing}")

    # 类型校验与转换
    type_check = {
        'user_id': 'object',
        'artist_id': 'object',
        'tag': 'object',
        'timestamp': 'object',
        'bias_type': 'object',
        'weight': 'float64'
    }

    for col, dtype in type_check.items():
        if df[col].dtype != dtype:
            print(f"警告: {col} 字段类型异常，正在转换...")
            df[col] = df[col].astype(dtype)

    # 保存处理结果
    df.to_csv(output_path, sep='\t', index=False)
    print(f"清洗完成，保存至: {output_path}")
    print(f"总记录数: {len(df)}")
    print(f"字段类型验证通过: {all(type_check[col] == df[col].dtype for col in required_columns)}")


def split_and_merge(raw_clean: pd.DataFrame, synthetic_dir: str, bias_type: str):
    """安全划分与合并数据"""
    # 输入验证
    if raw_clean.empty:
        raise ValueError(f"输入数据为空，无法处理偏差类型: {bias_type}")

    # 数据打乱
    shuffled = raw_clean.sample(frac=1, random_state=RANDOM_SEED)

    # 安全划分训练/验证集
    try:
        train_df, val_df = train_test_split(
            shuffled,
            test_size=0.2,
            random_state=RANDOM_SEED
        )
    except ValueError as e:
        print(f"数据划分失败: {str(e)}")
        return None, None

    # 合成数据加载
    syn_path = os.path.join(synthetic_dir, 'inter.tsv')
    if not os.path.exists(syn_path):
        print(f"警告: 合成数据文件不存在: {syn_path}")
        return train_df, val_df

    try:
        syn_df = pd.read_csv(syn_path, sep='\t')
    except Exception as e:
        print(f"合成数据加载失败: {str(e)}")
        return train_df, val_df

    # 字段验证
    required_cols = ['user_id', 'artist_id']
    for df in [train_df, val_df, syn_df]:
        if not set(required_cols).issubset(df.columns):
            missing = set(required_cols) - set(df.columns)
            raise ValueError(f"数据缺失必要字段 {missing}，无法合并。偏差类型: {bias_type}")

    # 合并数据（去重）
    merged_train = pd.concat([train_df, syn_df]).drop_duplicates(subset=required_cols)
    merged_val = pd.concat([val_df, syn_df]).drop_duplicates(subset=required_cols)

    # 最终验证
    if merged_train.empty or merged_val.empty:
        raise ValueError(f"合并后数据为空。训练集: {len(merged_train)}, 验证集: {len(merged_val)}")

    return merged_train, merged_val


def main():
    # 创建输出目录
    for bias in BIAS_TYPES:
        os.makedirs(os.path.join(PROCESSED_BASE_DIR, bias, 'train'), exist_ok=True)
        os.makedirs(os.path.join(PROCESSED_BASE_DIR, bias, 'val'), exist_ok=True)

    # 加载冷启动用户
    test_users = load_cold_start_users(TEST_USER_FILE)
    if not test_users:
        print("警告: 未检测到冷启动用户，使用全量数据")

    # 数据清洗流程
    try:
        clean_and_transform(
            RAW_JSON_PATH,
            CLEAN_OUTPUT,
            test_users
        )
    except Exception as e:
        print(f"数据清洗失败: {str(e)}")
        return

    # 处理每个偏差类型
    for bias_type in BIAS_TYPES:
        print(f"\n正在处理偏差类型: {bias_type}")
        synthetic_dir = SYNTHETIC_DIRS[bias_type]

        try:
            # 加载原始数据（已清洗）
            raw_clean = pd.read_csv(CLEAN_OUTPUT, sep='\t')

            # 划分合并数据
            train_df, val_df = split_and_merge(
                raw_clean.copy(),
                synthetic_dir,
                bias_type
            )

            # 保存结果
            train_path = os.path.join(PROCESSED_BASE_DIR, bias_type, 'train', 'inter.tsv')
            val_path = os.path.join(PROCESSED_BASE_DIR, bias_type, 'val', 'inter.tsv')

            train_df.to_csv(train_path, sep='\t', index=False)
            val_df.to_csv(val_path, sep='\t', index=False)

            print(f"偏差类型 {bias_type} 处理完成")
            print(f"训练集样本: {len(train_df)} | 验证集样本: {len(val_df)}")

        except Exception as e:
            print(f"偏差类型 {bias_type} 处理失败: {str(e)}")
            continue


if __name__ == '__main__':
    main()
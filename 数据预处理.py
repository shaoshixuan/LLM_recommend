import pandas as pd
import chardet
import os


def detect_encoding(file_path):
    """检测文件编码"""
    with open(file_path, 'rb') as f:
        result = chardet.detect(f.read(100000))  # 读取前100KB来检测编码
    return result['encoding']


def add_tag_id_column(behavior_file, tag_mapping_file, output_file, encoding=None):
    """
    将标签映射文件中的tag_id添加到行为数据文件中

    参数:
        behavior_file: 包含user_id、artist_id等信息的TSV文件路径
        tag_mapping_file: 包含tag和tag_id对应关系的TSV文件路径
        output_file: 输出文件路径，将在原数据基础上增加tag_id列
        encoding: 文件编码，默认为None(自动检测)
    """
    # 检查文件是否存在
    for file in [behavior_file, tag_mapping_file]:
        if not os.path.exists(file):
            raise FileNotFoundError(f"文件不存在: {file}")

    # 如果未指定编码，则自动检测
    if encoding is None:
        print("正在检测文件编码...")
        behavior_encoding = detect_encoding(behavior_file)
        mapping_encoding = detect_encoding(tag_mapping_file)

        print(f"行为文件编码: {behavior_encoding}")
        print(f"映射文件编码: {mapping_encoding}")

        # 如果两个文件编码不同，发出警告
        if behavior_encoding != mapping_encoding:
            print("警告: 两个文件的编码不同，可能会导致匹配问题")

        # 使用检测到的编码读取文件
        try:
            behavior_df = pd.read_csv(behavior_file, sep='\t', encoding=behavior_encoding)
            tag_mapping_df = pd.read_csv(tag_mapping_file, sep='\t', encoding=mapping_encoding)
        except UnicodeDecodeError:
            print(f"检测到的编码({behavior_encoding}/{mapping_encoding})无法正确解码文件")
            print("尝试使用常见的中文编码...")
            # 尝试常见的中文编码
            for enc in ['gbk', 'gb2312', 'utf-8-sig']:
                try:
                    print(f"尝试编码: {enc}")
                    behavior_df = pd.read_csv(behavior_file, sep='\t', encoding=enc)
                    tag_mapping_df = pd.read_csv(tag_mapping_file, sep='\t', encoding=enc)
                    print(f"成功使用编码: {enc}")
                    break
                except UnicodeDecodeError:
                    continue
            else:
                raise UnicodeDecodeError("无法确定正确的文件编码，请手动指定")
    else:
        # 使用指定的编码读取文件
        print(f"使用指定编码: {encoding}")
        behavior_df = pd.read_csv(behavior_file, sep='\t', encoding=encoding)
        tag_mapping_df = pd.read_csv(tag_mapping_file, sep='\t', encoding=encoding)

    # 确保标签映射文件中列名正确
    if 'tagID' not in tag_mapping_df.columns or 'tag' not in tag_mapping_df.columns:
        print("警告：标签映射文件的列名可能不符合预期")
        print(f"当前列名: {tag_mapping_df.columns.tolist()}")
        # 尝试重命名列
        if len(tag_mapping_df.columns) == 2:
            tag_mapping_df.columns = ['tag', 'tagID']
            print("已自动重命名列为: tag, tagID")

    # 关键修改：确保每个tag只对应一个tagID
    print(f"去重前标签映射文件行数: {len(tag_mapping_df)}")
    tag_mapping_df = tag_mapping_df.drop_duplicates(subset='tag')
    print(f"去重后标签映射文件行数: {len(tag_mapping_df)}")

    # 合并数据框
    merged_df = pd.merge(
        behavior_df,
        tag_mapping_df,
        on='tag',
        how='left'
    )

    # 检查是否有未匹配的标签
    unmatched_tags = merged_df[merged_df['tagID'].isna()]['tag'].unique()
    if len(unmatched_tags) > 0:
        print(f"警告：以下标签在映射文件中未找到匹配: {unmatched_tags}")

    # 保存结果，统一使用UTF-8编码
    merged_df.to_csv(output_file, sep='\t', index=False, encoding='utf-8')
    print(f"处理完成，结果已保存到 {output_file}")
    print(f"原始数据: {len(behavior_df)} 行, 处理后: {len(merged_df)} 行")


# 使用示例
if __name__ == "__main__":
    behavior_file = r"control_data.tsv"
    tag_mapping_file = r"F:\LR\dataset\processed\control\train\item.tsv"
    output_file = "control_data_yuchuli.tsv"  # 输出文件路径

    add_tag_id_column(behavior_file, tag_mapping_file, output_file)
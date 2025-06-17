import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_recall_curve
import os
import time
from tqdm import tqdm
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows


# 设置随机种子以确保结果可复现
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed()


# 1. 数据加载与预处理（读取两个文件：行为数据和tag映射）
def load_and_preprocess_data(behavior_file, tag_mapping_file):
    """
    加载行为数据和tag映射文件，返回处理后的数据和映射关系
    """
    # 读取行为数据，跳过timestamp和bias_type列
    df = pd.read_csv(behavior_file, sep='\t')
    df = df[['user_id', 'artist_id', 'tag', 'weight', 'tagID']]
    df = df.dropna()
    df=df.iloc[:20000]
    # 读取tag映射文件（tag到tagID的对应关系）
    for enc in ['gbk', 'gb2312', 'utf-8-sig']:
        try:
            print(f"尝试编码: {enc}")
            tag_mapping = pd.read_csv(tag_mapping_file, sep='\t', encoding=enc)
            print(f"成功使用编码: {enc}")
            break
        except UnicodeDecodeError:
            continue
    # 建立tagID到tag的映射字典
    tag_id_to_tag = dict(zip(tag_mapping['tagID'], tag_mapping['tag']))
    # 建立tag到tagID的映射字典（备用）
    tag_to_id = dict(zip(tag_mapping['tag'], tag_mapping['tagID']))

    # 创建用户-标签交互矩阵
    user_tag_matrix = df.pivot_table(
        index='user_id',
        columns='tagID',
        values='weight',
        aggfunc='sum',
        fill_value=0
    )

    # 获取特征数量（确保包含所有可能的ID）
    feature_counts = {
        'user': df['user_id'].max() + 1,
        'artist': df['artist_id'].max() + 1,
        'tag': df['tagID'].max() + 1
    }

    # 检查数据有效性
    if df.isna().any().any():
        print("警告：数据中存在NaN值，将进行填充")
        df = df.fillna(0)

    return df, user_tag_matrix, feature_counts, tag_id_to_tag, tag_to_id


# 2. 构建序列数据（使用行为数据中的自然顺序）
def build_sequences(df, seq_len=5):
    """构建用户行为序列数据（基于原始数据的自然顺序）"""
    # 按用户ID分组，假设数据已按用户行为顺序排列
    user_sequences = {}
    for user_id, group in df.groupby('user_id'):
        tags = group['tagID'].values  # 使用tagID
        artists = group['artist_id'].values
        weights = group['weight'].values

        # 创建序列
        sequences = []
        for i in range(len(tags) - seq_len):
            # 历史行为序列
            hist_tags = tags[i:i + seq_len]
            hist_artists = artists[i:i + seq_len]
            hist_weights = weights[i:i + seq_len]

            # 目标标签
            target_tag = tags[i + seq_len]
            target_artist = artists[i + seq_len]

            sequences.append({
                'hist_tags': hist_tags,
                'hist_artists': hist_artists,
                'hist_weights': hist_weights,
                'target_tag': target_tag,
                'target_artist': target_artist,
                'user_id': user_id,
                'label': 1  # 正样本
            })

        user_sequences[user_id] = sequences

    # 合并所有用户的序列
    all_sequences = []
    for seq_list in user_sequences.values():
        all_sequences.extend(seq_list)

    return pd.DataFrame(all_sequences)


# 3. 负采样（确保负样本ID有效）
def negative_sampling(sequences_df, df, tag_to_id, feature_counts, neg_ratio=3):
    """为序列数据生成负样本（确保tagID在有效范围内）"""
    # 获取所有有效的tagID（确保不超过嵌入层大小）
    max_tag_id = feature_counts['tag'] - 1
    all_valid_tag_ids = [tag_id for tag_id in tag_to_id.values() if tag_id <= max_tag_id]

    negative_samples = []
    for _, row in sequences_df.iterrows():
        user_id = row['user_id']
        target_tag_id = row['target_tag']

        # 获取当前用户已交互的tagID
        user_tag_ids = df[df['user_id'] == user_id]['tagID'].unique()
        # 过滤出有效且未交互的tagID
        neg_candidates = np.setdiff1d(all_valid_tag_ids, user_tag_ids)

        if len(neg_candidates) > 0:
            # 随机选择负样本
            neg_tag_ids = np.random.choice(
                neg_candidates,
                min(neg_ratio, len(neg_candidates)),
                replace=False
            )

            for neg_tag_id in neg_tag_ids:
                negative_samples.append({
                    'hist_tags': row['hist_tags'],
                    'hist_artists': row['hist_artists'],
                    'hist_weights': row['hist_weights'],
                    'target_tag': neg_tag_id,
                    'target_artist': row['target_artist'],
                    'user_id': user_id,
                    'label': 0  # 负样本
                })

    # 合并正样本和负样本
    return pd.concat([sequences_df, pd.DataFrame(negative_samples)], ignore_index=True)


# 4. 数据集类
class DINDataset(Dataset):
    """DIN模型的数据集类"""

    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        hist_tags = torch.tensor(row['hist_tags'], dtype=torch.long)
        hist_artists = torch.tensor(row['hist_artists'], dtype=torch.long)
        hist_weights = torch.tensor(row['hist_weights'], dtype=torch.float)

        target_tag = torch.tensor(row['target_tag'], dtype=torch.long)
        target_artist = torch.tensor(row['target_artist'], dtype=torch.long)

        user_id = torch.tensor(row['user_id'], dtype=torch.long)
        label = torch.tensor(row['label'], dtype=torch.float)

        # 检查数据有效性
        if torch.isnan(hist_weights).any() or torch.isinf(hist_weights).any():
            hist_weights = torch.zeros_like(hist_weights)

        return {
            'hist_tags': hist_tags,
            'hist_artists': hist_artists,
            'hist_weights': hist_weights,
            'target_tag': target_tag,
            'target_artist': target_artist,
            'user_id': user_id,
            'label': label
        }


# 5. DIN模型的激活单元
class AttentionUnit(nn.Module):
    """DIN模型的注意力单元"""

    def __init__(self, embedding_dim, hidden_dims=[80, 40]):
        super(AttentionUnit, self).__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dims = hidden_dims

        # 构建MLP层
        layers = []
        input_dim = embedding_dim * 4  # query, key, query-key, query*key

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.PReLU())
            input_dim = hidden_dim

        layers.append(nn.Linear(input_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, query, keys, keys_length):
        """
        query: 目标商品的embedding, [B, E]
        keys: 历史行为的embedding, [B, T, E]
        keys_length: 历史行为序列的长度, [B]
        """
        batch_size, seq_len, embedding_dim = keys.size()

        # 扩展query的维度以匹配keys
        query = query.unsqueeze(1).expand(-1, seq_len, -1)  # [B, T, E]

        # 计算query和key之间的差异
        att_input = torch.cat([
            query, keys, query - keys, query * keys
        ], dim=-1)  # [B, T, 4*E]

        # 通过MLP计算注意力权重
        att_score = self.mlp(att_input).squeeze(-1)  # [B, T]

        # 掩码处理
        mask = (torch.arange(seq_len, device=keys.device).expand(batch_size, seq_len) <
                keys_length.unsqueeze(1))  # [B, T]

        # 应用掩码并归一化
        att_score = att_score.masked_fill(~mask, -1e9)
        att_score = F.softmax(att_score, dim=1)  # [B, T]

        # 应用注意力权重
        att_score = att_score.unsqueeze(-1)  # [B, T, 1]
        output = torch.sum(att_score * keys, dim=1)  # [B, E]

        return output


# 6. DIN模型
class DIN(nn.Module):
    """深度兴趣网络模型"""

    def __init__(self,
                 user_count,
                 tag_count,
                 artist_count,
                 embedding_dim=64,
                 hidden_dims=[128, 64]):
        super(DIN, self).__init__()

        # 嵌入层（减小默认维度以降低内存占用）
        self.user_embedding = nn.Embedding(user_count, embedding_dim)
        self.tag_embedding = nn.Embedding(tag_count, embedding_dim)
        self.artist_embedding = nn.Embedding(artist_count, embedding_dim)

        # 注意力单元
        self.attention = AttentionUnit(embedding_dim)

        # 全连接层（减小隐藏层维度）
        layers = []
        input_dim = embedding_dim * 4  # user, target_tag, target_artist, hist_attention

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.PReLU())
            input_dim = hidden_dim

        layers.append(nn.Linear(input_dim, 1))
        self.mlp = nn.Sequential(*layers)

        # 初始化参数
        self._init_weights()

    def _init_weights(self):
        """初始化模型参数"""
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0, std=0.01)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, user_id, target_tag, target_artist, hist_tags, hist_artists, hist_weights, seq_lengths):
        """前向传播"""
        # 获取嵌入向量
        user_emb = self.user_embedding(user_id)  # [B, E]
        target_tag_emb = self.tag_embedding(target_tag)  # [B, E]
        target_artist_emb = self.artist_embedding(target_artist)  # [B, E]

        # 获取历史行为的嵌入
        hist_tag_emb = self.tag_embedding(hist_tags)  # [B, T, E]
        hist_artist_emb = self.artist_embedding(hist_artists)  # [B, T, E]

        # 组合历史行为的嵌入
        hist_emb = hist_tag_emb + hist_artist_emb  # [B, T, E]

        # 应用注意力机制
        hist_attention = self.attention(target_tag_emb, hist_emb, seq_lengths)  # [B, E]

        # 拼接所有特征
        concat_features = torch.cat([
            user_emb,
            target_tag_emb,
            target_artist_emb,
            hist_attention
        ], dim=1)  # [B, 4*E]

        # 通过MLP预测
        output = self.mlp(concat_features).squeeze(-1)  # [B]
        output = torch.sigmoid(output)

        return output


# 7. 评估函数
def calculate_metrics(y_true, y_score, threshold=0.5):
    """计算多种评估指标"""
    y_pred = (y_score > threshold).astype(int)

    # 基本指标
    auc = roc_auc_score(y_true, y_score)
    accuracy = accuracy_score(y_true, y_pred)

    # F1分数（处理二分类情况）
    try:
        f1 = f1_score(y_true, y_pred)
    except:
        # 处理全0或全1的情况
        f1 = 0.0 if np.sum(y_true) == 0 else 1.0

    # 精确率和召回率
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    pr_auc = -np.trapz(recall, precision)  # 注意：这里需要取负，因为precision_recall_curve返回的是逆序的

    return {
        'AUC': auc,
        'F1': f1,
        'Accuracy': accuracy,
        'PR_AUC': pr_auc,
        'Precision': precision[1] if len(precision) > 1 else 0,  # 第一个有效点
        'Recall': recall[1] if len(recall) > 1 else 0
    }


# 8. 训练函数（增加评估指标）
def train_model(model, train_dataset, val_dataset, epochs=10, lr=0.001, device='cuda', initial_batch_size=128):
    """训练DIN模型（动态调整批量大小并计算评估指标）"""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    val_metrics = []

    # 尝试降低批量大小直到找到合适的值
    current_batch_size = initial_batch_size

    while True:
        try:
            # 重新创建DataLoader对象，使用新的batch_size
            train_loader = DataLoader(train_dataset, batch_size=current_batch_size, shuffle=True, num_workers=4)
            val_loader = DataLoader(val_dataset, batch_size=current_batch_size, shuffle=False, num_workers=4)

            print(f"使用批量大小: {current_batch_size}")

            for epoch in range(epochs):
                # 训练阶段
                model.train()
                train_loss = 0.0

                for batch in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs} [Train], BS: {current_batch_size}'):
                    # 将数据移至设备
                    user_id = batch['user_id'].to(device)
                    target_tag = batch['target_tag'].to(device)
                    target_artist = batch['target_artist'].to(device)
                    hist_tags = batch['hist_tags'].to(device)
                    hist_artists = batch['hist_artists'].to(device)
                    hist_weights = batch['hist_weights'].to(device)
                    labels = batch['label'].to(device)

                    # 计算序列长度
                    seq_lengths = torch.sum(hist_weights > 0, dim=1).to(device)

                    # 前向传播
                    optimizer.zero_grad()
                    outputs = model(user_id, target_tag, target_artist, hist_tags, hist_artists, hist_weights,
                                    seq_lengths)

                    # 计算损失
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    train_loss += loss.item()

                train_loss /= len(train_loader)
                train_losses.append(train_loss)

                # 验证阶段
                model.eval()
                val_loss = 0.0
                all_val_labels = []
                all_val_scores = []

                with torch.no_grad():
                    for batch in tqdm(val_loader, desc=f'Epoch {epoch + 1}/{epochs} [Val], BS: {current_batch_size}'):
                        # 将数据移至设备
                        user_id = batch['user_id'].to(device)
                        target_tag = batch['target_tag'].to(device)
                        target_artist = batch['target_artist'].to(device)
                        hist_tags = batch['hist_tags'].to(device)
                        hist_artists = batch['hist_artists'].to(device)
                        hist_weights = batch['hist_weights'].to(device)
                        labels = batch['label'].to(device)

                        # 计算序列长度
                        seq_lengths = torch.sum(hist_weights > 0, dim=1).to(device)

                        # 前向传播
                        outputs = model(user_id, target_tag, target_artist, hist_tags, hist_artists, hist_weights,
                                        seq_lengths)

                        # 计算损失
                        loss = criterion(outputs, labels)
                        val_loss += loss.item()

                        # 收集标签和分数用于评估
                        all_val_labels.extend(labels.cpu().numpy())
                        all_val_scores.extend(outputs.cpu().numpy())

                val_loss /= len(val_loader)
                val_losses.append(val_loss)

                # 计算评估指标
                if all_val_labels:
                    metrics = calculate_metrics(np.array(all_val_labels), np.array(all_val_scores))
                    val_metrics.append(metrics)
                    print(f'Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, '
                          f'AUC: {metrics["AUC"]:.4f}, F1: {metrics["F1"]:.4f}, Accuracy: {metrics["Accuracy"]:.4f}')
                else:
                    print(f'Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, '
                          f'评估指标: 验证集数据不足')

                # 保存最佳模型
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(model.state_dict(), 'best_din_model.pth')
                    print(f'Saved best model with val loss: {best_val_loss:.4f}')

            break  # 训练成功，退出循环
        except RuntimeError as e:
            if 'CUDA' in str(e) and 'CUBLAS' in str(e):
                # 检测到CUDA错误，尝试减小批量大小
                current_batch_size = max(16, current_batch_size // 2)
                print(f"检测到CUDA内存问题，减小批量大小到 {current_batch_size}")
            else:
                raise e

    return train_losses, val_losses, val_metrics


# 9. 推荐函数（带tagID到tag名称的转换，增加批量处理能力）
def recommend_for_users(model, user_ids, all_tags, all_artists, user_histories, feature_counts, tag_id_to_tag,
                        top_n=10, device='cuda', batch_size=50):
    """为多个用户批量生成推荐，并将tagID转换为tag名称"""
    model.eval()
    all_recommendations = []

    # 分批处理用户以避免内存问题
    for i in range(0, len(user_ids), batch_size):
        batch_user_ids = user_ids[i:i + batch_size]

        for user_id in batch_user_ids:
            # 准备用户历史数据
            if user_id not in user_histories:
                continue

            user_history = user_histories[user_id]
            hist_tags = user_history['hist_tags']
            hist_artists = user_history['hist_artists']
            hist_weights = user_history['hist_weights']

            # 优化：直接从numpy数组创建张量，避免中间列表转换
            hist_tags_tensor = torch.from_numpy(hist_tags).unsqueeze(0).to(device)
            hist_artists_tensor = torch.from_numpy(hist_artists).unsqueeze(0).to(device)
            hist_weights_tensor = torch.from_numpy(hist_weights).unsqueeze(0).to(device)
            seq_lengths = torch.sum(hist_weights_tensor > 0, dim=1).to(device)

            # 对每个标签计算预测分数
            scores = []
            for tag_id in all_tags:
                # 确保tagID有效
                if tag_id >= feature_counts['tag']:
                    continue

                # 对每个标签选择一个代表性的艺术家
                artist_id = all_artists[0]  # 简化处理，实际中可以选择与标签最相关的艺术家

                # 优化：直接从整数创建张量
                target_tag_tensor = torch.tensor([tag_id], dtype=torch.long).to(device)
                target_artist_tensor = torch.tensor([artist_id], dtype=torch.long).to(device)
                user_id_tensor = torch.tensor([user_id], dtype=torch.long).to(device)

                # 预测
                with torch.no_grad():
                    score = model(
                        user_id_tensor,
                        target_tag_tensor,
                        target_artist_tensor,
                        hist_tags_tensor,
                        hist_artists_tensor,
                        hist_weights_tensor,
                        seq_lengths
                    ).item()

                scores.append((tag_id, score))

            # 按分数排序并返回前N个
            scores.sort(key=lambda x: x[1], reverse=True)
            top_recommendations = scores[:top_n]

            # 转换tagID为tag名称并保存
            for tag_id, score in top_recommendations:
                tag_name = tag_id_to_tag.get(tag_id, f"未知标签({tag_id})")
                all_recommendations.append({
                    'user_id': user_id,
                    'tag_name': tag_name,
                    'tag_id': tag_id,
                    'score': score
                })

    return all_recommendations


# 10. 将推荐结果保存到Excel
def save_recommendations_to_excel(recommendations, output_file='user_recommendations.xlsx'):
    """将推荐结果保存到Excel文件"""
    if not recommendations:
        print("没有推荐结果可保存")
        return

    # 转换为DataFrame
    df = pd.DataFrame(recommendations)

    # 创建Excel文件
    wb = Workbook()
    ws = wb.active

    # 写入表头
    ws.append(df.columns.tolist())

    # 写入数据
    for row in dataframe_to_rows(df, index=False, header=False):
        ws.append(row)

    # 按用户ID分组创建不同工作表
    user_groups = df.groupby('user_id')
    for user_id, group in user_groups:
        ws_user = wb.create_sheet(title=f'用户_{user_id}')
        ws_user.append(group.columns.tolist())
        for row in dataframe_to_rows(group, index=False, header=False):
            ws_user.append(row)

    # 保存文件
    wb.save(output_file)
    print(f"推荐结果已保存到 {output_file}")


# 11. 主函数
def main():
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # 检查CUDA和PyTorch版本兼容性
    if device.type == 'cuda':
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"PyTorch版本: {torch.__version__}")
        print(f"GPU型号: {torch.cuda.get_device_name(0)}")

    # 替换为实际文件路径
    behavior_file = r'control_data_yuchuli.tsv'  # 行为数据文件
    tag_mapping_file = r'F:\LR\dataset\processed\control\train\item.tsv'  # tag映射文件

    # 加载数据
    print('Loading and preprocessing data...')
    df, user_tag_matrix, feature_counts, tag_id_to_tag, tag_to_id = load_and_preprocess_data(
        behavior_file, tag_mapping_file
    )

    # 构建序列数据
    print('Building sequences...')
    seq_len = 3  # 减小序列长度以降低内存占用
    sequences_df = build_sequences(df, seq_len)

    # 负采样（传递feature_counts确保负样本ID有效）
    print('Performing negative sampling...')
    neg_ratio = 2  # 减小负样本比例
    train_data = negative_sampling(sequences_df, df, tag_to_id, feature_counts, neg_ratio)

    # 分割训练集和测试集
    print('Splitting data into train and test sets...')
    train_df, test_df = train_test_split(train_data, test_size=0.2, random_state=42)

    # 创建数据集
    train_dataset = DINDataset(train_df)
    test_dataset = DINDataset(test_df)

    initial_batch_size = 128  # 初始批量大小

    # 初始化模型（减小嵌入维度和隐藏层维度）
    print('Initializing DIN model...')
    model = DIN(
        user_count=feature_counts['user'],
        tag_count=int(feature_counts['tag']),
        artist_count=int(feature_counts['artist']),
        embedding_dim=64,
        hidden_dims=[128, 64]
    )

    # 训练模型（使用数据集而非DataLoader，增加评估指标）
    print('Training model...')
    epochs = 10
    lr = 0.001
    train_losses, val_losses, val_metrics = train_model(
        model=model,
        train_dataset=train_dataset,
        val_dataset=test_dataset,
        epochs=epochs,
        lr=lr,
        device=device,
        initial_batch_size=initial_batch_size
    )

    # 打印最佳评估指标
    if val_metrics:
        best_epoch = np.argmin(val_losses)
        print(f"\n最佳模型在第 {best_epoch + 1} 轮训练中，评估指标:")
        for metric_name, value in val_metrics[best_epoch].items():
            print(f"{metric_name}: {value:.4f}")

    # 加载最佳模型
    model.load_state_dict(torch.load('best_din_model.pth'))

    # 为用户生成推荐并保存到Excel
    print('Generating recommendations and saving to Excel...')
    top_n = 5  # 每个用户推荐的标签数量

    # 准备数据
    all_tags = [tag_id for tag_id in range(feature_counts['tag'])]
    all_artists = [artist_id for artist_id in range(feature_counts['artist'])]

    # 构建用户历史字典（提高查询效率）
    user_histories = {}
    for user_id, group in train_data.groupby('user_id'):
        if len(group) > 0:
            user_histories[user_id] = group.iloc[0].to_dict()

    # 选择要推荐的用户（这里选择所有有历史记录的用户）
    recommend_user_ids = list(user_histories.keys())[:5]  # 限制推荐用户数量，可根据需要调整

    # 生成推荐
    recommendations = recommend_for_users(
        model=model,
        user_ids=recommend_user_ids,
        all_tags=all_tags,
        all_artists=all_artists,
        user_histories=user_histories,
        feature_counts=feature_counts,
        tag_id_to_tag=tag_id_to_tag,
        top_n=top_n,
        device=device
    )

    # 保存到Excel
    save_recommendations_to_excel(recommendations, 'user_recommendations.xlsx')

    print('Done!')


if __name__ == "__main__":
    main()
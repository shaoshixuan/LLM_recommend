import pandas as pd
#Part1：数据处理
# 1. 加载原始数据
user_artists = pd.read_csv('F:/LR/hetrec2011-lastfm-2k/user_artists.dat', sep='\t', encoding='latin-1')
user_tags = pd.read_csv('F:/LR/hetrec2011-lastfm-2k/user_taggedartists-timestamps.dat', sep='\t', encoding='latin-1')
tags = pd.read_csv('F:/LR/hetrec2011-lastfm-2k/tags.dat', sep='\t', encoding='latin-1')
artists = pd.read_csv('F:/LR/hetrec2011-lastfm-2k/artists.dat', sep='\t', encoding='latin-1')

user_tags['datetime'] =pd.to_datetime(user_tags['timestamp'], unit='ms')  # 转换为datetime格式

# 2. 新表1：合并标签信息 + 艺术家信息
# （1）将标签ID映射为标签名称
user_tags_merged = pd.merge(user_tags, tags, on='tagID', how='left')

# （2）再将艺术家ID映射为艺术家名称
# user_tags_final = pd.merge(user_tags_merged, artists[['id', 'name']], left_on='artistID', right_on='id', how='left')
user_tag_table = user_tags_merged[['userID', 'artistID','tagValue','timestamp','datetime']]

# 3. 新表2：合并听歌行为 + 艺术家名称
df_merged = pd.merge(user_artists,user_tag_table,on=['userID','artistID'],how='left')

df_merged = df_merged.sort_values(by=['userID','timestamp'])
df_merged = df_merged[['userID', 'artistID', 'weight', 'tagValue', 'timestamp', 'datetime']]
# print(df_merged)
df_merged.to_json('D:/user_behavior_sequence.json', orient='records', lines=True)
# 获取用户-艺术家为单位的行为序列
import pandas as pd
import json

# 2. 确保 timestamp 是整数格式

# 先删除 timestamp 中为 NaN 或无穷的行
df_merged = df_merged[pd.to_numeric(df_merged['timestamp'], errors='coerce').notnull()]  # 删除无法转换为数字的行
df_merged = df_merged[~df_merged['timestamp'].isin([float('inf'), float('-inf')])]      # 删除正负无穷
df_merged['timestamp'] = df_merged['timestamp'].astype(int)

# 3. 分组并提取 initial、end、peak 三个认知偏差特征
grouped = df_merged.groupby(['userID', 'artistID'])

result = []

for (user, artist), group in grouped:
    group_sorted = group.sort_values(by='timestamp')

    # initial
    initial_row = group_sorted.iloc[0]
    initial_timestamp = initial_row['timestamp']
    initial_tag = initial_row['tagValue']

    # end
    end_row = group_sorted.iloc[-1]
    end_timestamp = end_row['timestamp']
    end_tag = end_row['tagValue']

    # high (最大 weight 的记录)
    high_row = group_sorted.loc[group_sorted['weight'].idxmax()]
    high_timestamp = high_row['timestamp']
    high_tag = high_row['tagValue']

    result.append({
        'userID': user,
        'artistID': artist,
        'weight': high_row['weight'],  # 假定 peak 的行为是代表性的
        'initial_timestamp': initial_timestamp,
        'initial_tagValue': initial_tag,
        'end_timestamp': end_timestamp,
        'end_tagValue': end_tag,
        'high_timestamp': high_timestamp,
        'high_tagValue': high_tag
    })

# 4. 构建最终 DataFrame
result_df = pd.DataFrame(result)

# 5. 保存为 JSON 行格式
# result_df.to_json('D:/final_user_bias_data.json', orient='records', lines=True, force_ascii=False)

# 6. 如果需要预览前几行

# 统计每个用户的行为数量
user_behavior_counts = result_df.groupby('userID').size().reset_index(name='behavior_count')

# 设置阈值：例如 <= 5 条行为为冷启动用户
cold_start_threshold = 5

# 筛选冷启动用户
cold_start_users = user_behavior_counts[user_behavior_counts['behavior_count'] <= cold_start_threshold]['userID']

# 进一步筛选伪冷启动用户（行为数较多的活跃用户）
# 例如行为数 >= 20 的用户
pseudo_candidates = user_behavior_counts[user_behavior_counts['behavior_count'] >= 50]['userID']

# 将伪冷启动用户行为数据截断为前 N 条
N = 5  # 截断行为数量
pseudo_cold_start_behaviors = (
    result_df[result_df['userID'].isin(pseudo_candidates)]
    .sort_values(['userID', 'initial_timestamp'])
    .groupby('userID')
    .head(N)
)

# 获取冷启动用户行为数据
cold_start_behaviors = result_df[result_df['userID'].isin(cold_start_users)]

# 合并冷启动和伪冷启动样本
combined_cold_start_data = pd.concat([cold_start_behaviors, pseudo_cold_start_behaviors])
combined_cold_start_data.to_json('D:/cold_start_data.json', orient='records', force_ascii=False)
print(combined_cold_start_data.head())
import json
import random
import pandas as pd
from datetime import datetime

# 载入所有 artistID
artist_ids = artists['id'].tolist()  # 转换为列表
# Last.fm数据集时间范围
LASTFM_MIN_TS = 1104537600  # 2005-01-01
LASTFM_MAX_TS = 1246406400  # 2009-07-01


# 转换为可读日期
min_date = datetime.utcfromtimestamp(LASTFM_MIN_TS).strftime('%Y-%m-%d')
max_date = datetime.utcfromtimestamp(LASTFM_MAX_TS).strftime('%Y-%m-%d')

min_timestamp = LASTFM_MIN_TS
max_timestamp = LASTFM_MAX_TS
# [{"userID":"用户ID", "artistID":"艺术家ID", "tagValue":"风格标签", "weight":"听歌次数", "timestamp":"标准时间戳"},...]
# 四种认知偏差的prompt模板 - 使用描述性语言体现偏差
PROMPT_TEMPLATES = {
    "control": {
        "instruction": "你是一个模拟用户行为的专家，请根据下面冷启动用户{user_id}的偏好数据，生成一个真实且合理的音乐行为序列（行为数量为5条）用于训练推荐系统.",
        "bias_note": "用户行为遵循自然偏好分布，不受特定认知偏差影响。行为应反映真实用户的随机探索模式，不强调特定偏好模式。",
        "output_fields":["userID", "artistID", "tagValue", "weight", "timestamp"]
    },
    "anchor": {
        "instruction": "你是一个模拟用户行为的专家，请根据下面冷启动用户{user_id}的偏好数据，生成一个真实且合理的音乐行为序列（行为数量为5条）用于训练推荐系统.",
        "bias_note": "用户的行为会更加集中于他最早听过的风格（即{initial_tagValue}）,其他风格可以作为补充，但比例较低",
        "output_fields":["userID", "artistID", "tagValue", "weight", "timestamp"]
    },
    "peak-end": {
        "instruction": "你是一个模拟用户行为的专家，请根据下面冷启动用户{user_id}的偏好数据，生成一个真实且合理的音乐行为序列（行为数量为5条）用于训练推荐系统.",
        "bias_note": "用户的行为会更加集中于听歌高峰体验和最近一次的偏好（即{high_tagValue}和{end_tagValue})，并且高峰体验时听歌次数为{weight},在生成过程中应强化这两个风格在听歌行为序列中的比例.",
        "output_fields": ["userID", "artistID", "tagValue", "weight", "timestamp"]
    },
    "both": {
        "instruction": "你是一个模拟用户行为的专家，请根据下面冷启动用户{user_id}的偏好数据，生成一个真实且合理的音乐行为序列（行为数量为5条）用于训练推荐系统.",
        "bias_note": "用户行为同时受初始锚点和峰终体验影响。想象这位用户既被初次'{initial_tagValue}'风格吸引，对于高峰时'{high_tagValue}'的风格和最近一次'{end_tagValue}'的风格记忆深刻，并且高峰体验时听歌次数为{weight}。",
        "output_fields": ["userID", "artistID", "tagValue", "weight", "timestamp"]
    }
}


def generate_prompt(user_data: dict, bias_type: str, artist_id_hint: str) -> str:
    """生成特定偏差类型的prompt"""
    template = PROMPT_TEMPLATES[bias_type]

    # 使用描述性语言构建提示
    prompt = f"""### 用户行为模拟任务
{template['instruction']}

## 关键约束
1. artistID必须严格从下列300个候选ID中选择（禁止使用列表外ID）:{artist_id_hint}
2. 时间戳规范（基于Last.fm数据集范围）：
   - 所有时间戳需在{min_timestamp}({min_date})到{max_timestamp}({max_date})范围内
   - 使用Unix时间戳格式
3. 标签值要求：
   - 使用Last.fm标准音乐流派标签如"rock", "jazz", "electronic"
   - 保持风格演变合理
   - 标签值应该反映用户对该artist的偏好变化

## 认知行为模式
{template['bias_note'].format(**user_data)}
"""

    # JSON格式规范
    fields = ", ".join(template["output_fields"])
    format_spec = f"""
## 输出要求
- 格式：JSON数组（包含5个元素）
- 字段顺序：{fields}
- 时间顺序：行为按时间升序排列
"""
    return prompt + format_spec


def batch_generate_prompts(user_df: pd.DataFrame) -> dict:
    """批量生成用户prompt"""
    # 随机选择用户样本
    # 为每个用户生成四种偏差的prompt
    results = {}
    for _, user in user_df.iterrows():
        user_data = user.to_dict()
        artist_id_sample = random.sample(artist_ids, min(300, len(artist_ids)))
        artist_id_hint = ", ".join(map(str, artist_id_sample))

        user_prompts = {}
        for bias in PROMPT_TEMPLATES.keys():
            user_prompts[bias] = generate_prompt(user_data, bias, artist_id_hint)

        results[user_data['userID']] = {
            "data": user_data,
            "prompts": user_prompts
        }

    return results


import json
import time
import random
import pandas as pd
import requests
import re
from datetime import datetime
from tenacity import retry, stop_after_attempt, wait_random_exponential, retry_if_exception_type

# 硅基流API配置
api_url = 'https://api.siliconflow.cn/v1/chat/completions'
api_key = 'sk-ccijjkwlpcdayaephradpxvlvibngjhelmhgqsjiefbpuioi'
model = 'deepseek-ai/DeepSeek-R1-Distill-Qwen-32B'


# 增强的重试机制
@retry(
    stop=stop_after_attempt(5),
    wait=wait_random_exponential(min=3, max=60),
    retry=(retry_if_exception_type(requests.exceptions.Timeout) |
           retry_if_exception_type(requests.exceptions.ConnectionError)))

def generate_response_sf(prompt: str) -> str:
    """调用硅基流API获取响应（增强版）"""
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    data = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "max_tokens": 2000
    }

    try:
        response = requests.post(api_url, headers=headers, json=data, timeout=(15, 240))
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            error_msg = f"API错误: {response.status_code} - {response.text[:200]}"
            print(error_msg)
            raise requests.exceptions.RequestException(error_msg)
    except requests.exceptions.Timeout:
        print("请求超时，重试中...")
        raise
    except requests.exceptions.RequestException as e:
        print(f"请求异常: {str(e)[:200]}")
        raise


def parse_behavior_response(response: str, user_id: int) :
    """解析API返回的行为数据（增强版）"""
    if response is None:
        print("响应为None，无法解析")
        return []

    # 尝试直接解析JSON
    try:
        start_idx = response.find('[')
        end_idx = response.rfind(']') + 1
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            json_str = response[start_idx:end_idx]
            parsed = json.loads(json_str)
        else:
            parsed = json.loads(response)

        # 标准化为列表
        if not isinstance(parsed, list):
            parsed = [parsed] if parsed else []

        behaviors = []
        for item in parsed:
            # 如果是字典，直接处理
            if isinstance(item, dict):
                behaviors.append(item)
            # 如果是列表，转换为字典
            elif isinstance(item, list):
                # 假设格式为 [user_id, artistID, tagValue, weight, timestamp]
                if len(item) >= 4:  # 至少需要4个字段
                    behaviors.append({
                        "userID": user_id if item[0] == '{user_id}' else item[0],
                        "artistID": item[1],
                        "tagValue": item[2],
                        "weight": item[3],
                        "timestamp": item[4] if len(item) > 4 else random.randint(LASTFM_MIN_TS, LASTFM_MAX_TS)
                    })
                else:
                    print(f"  无效的行为记录（列表长度不足）: {item}")
                    continue
            else:
                print(f"  无效的行为记录（格式未知）: {item}")
                continue
        return behaviors

    except json.JSONDecodeError:
        print("JSON解析失败，尝试修复...")

        try:
            # 处理单引号、True/False等问题
            fixed = response.replace("'", '"').replace("True", "true").replace("False", "false")
            fixed = re.sub(r'(\w+):\s*([^",{]+)([,}])', r'"\1": "\2"\3', fixed)
            parsed = json.loads(fixed)
            if not isinstance(parsed, list):
                parsed = [parsed] if parsed else []
            behaviors = []
            for item in parsed:
                if isinstance(item, dict):
                    behaviors.append(item)
                elif isinstance(item, list) and len(item) >= 4:
                    behaviors.append({
                        "userID": user_id if item[0] == '{user_id}' else item[0],
                        "artistID": item[1],
                        "tagValue": item[2],
                        "weight": item[3],
                        "timestamp": item[4] if len(item) > 4 else random.randint(LASTFM_MIN_TS, LASTFM_MAX_TS)
                    })
                else:
                    print(f"  无效的行为记录（修复后仍无效）: {item}")
                    continue
            return behaviors
        except json.JSONDecodeError:
            print("修复JSON失败，尝试提取部分数据...")

            pattern = r'\[.*?\]|\{.*?\}'
            matches = re.findall(pattern, response, re.DOTALL)
            behaviors = []
            for match in matches:
                try:
                    obj = json.loads(match)
                    if isinstance(obj, dict):
                        behaviors.append(obj)
                    elif isinstance(obj, list) and len(obj) >= 4:
                        behaviors.append({
                            "userID": user_id if obj[0] == '{user_id}' else obj[0],
                            "artistID": obj[1],
                            "tagValue": obj[2],
                            "weight": obj[3],
                            "timestamp": obj[4] if len(obj) > 4 else random.randint(LASTFM_MIN_TS, LASTFM_MAX_TS)
                        })
                    else:
                        print(f"  无效的行为记录（提取后仍无效）: {obj}")
                except:
                    continue
            print(f"  成功提取 {len(behaviors)} 条记录")
            return behaviors
    except Exception as e:
        print(f"解析行为数据时发生未知错误: {str(e)}")
        return []


def generate_behavior_data(user_df: pd.DataFrame, output_file: str = "generated_behaviors.json") :
    """
    批量生成用户行为数据（增强版）
    """
    print("生成用户prompt...")
    all_prompts = batch_generate_prompts(user_df)  # 假设此函数已定义
    print(f"已为 {len(all_prompts)} 位用户生成prompt")

    results = {}
    total_success = 0
    total_failed = 0

    for user_id, data in all_prompts.items():
        print(f"\n{'=' * 50}\n处理用户: {user_id}")
        user_results = {}

        bias_types = list(PROMPT_TEMPLATES.keys())
        for bias_type in bias_types:
            print(f"\n>> 生成 {bias_type} 偏差行为 <<")
            prompt = data['prompts'][bias_type]

            try:
                start_time = time.time()
                response = generate_response_sf(prompt)
                elapsed = time.time() - start_time

                if response is None:
                    print(f"  获取响应失败 ({elapsed:.1f}s)")
                    user_results[bias_type] = []
                    total_failed += 1
                    continue

                print(f"  获取响应成功 ({elapsed:.1f}s)")

                # 解析响应，传入 user_id 以替换占位符
                behavior_data = parse_behavior_response(response, user_id)
                print(f"  解析到 {len(behavior_data)} 条行为记录")

                # 验证和标准化数据
                valid_data = []
                for item in behavior_data:
                    if not isinstance(item, dict):
                        print(f"  无效的行为记录（非字典）: {item}")
                        continue

                    # 确保所有字段存在并标准化
                    item['userID'] = user_id  # 强制设置 userID
                    item['bias_type'] = bias_type

                    for field in ['artistID', 'tagValue', 'weight', 'timestamp']:
                        if field not in item or item[field] is None:
                            if field == 'weight':
                                item[field] = random.randint(1000, 10000)
                            elif field == 'timestamp':
                                item[field] = random.randint(LASTFM_MIN_TS, LASTFM_MAX_TS)
                            else:
                                item[field] = "未知"

                    # 处理权重
                    try:
                        weight = item['weight']
                        if isinstance(weight, str):
                            if '%' in weight:
                                weight = float(weight.strip('%')) / 100.0
                            else:
                                weight = float(weight)
                        if weight < 100:
                            weight = int(weight * 1000)
                        if weight < 1000:
                            weight = random.randint(1000, 10000)
                        elif weight > 50000:
                            weight = min(weight, 20000)
                        item['weight'] = int(weight)
                    except (ValueError, TypeError):
                        print(f"  无效的权重值: {item['weight']}，使用默认值")
                        item['weight'] = random.randint(1000, 10000)

                    # 验证时间戳
                    try:
                        ts = int(item['timestamp'])
                        if not (LASTFM_MIN_TS <= ts <= LASTFM_MAX_TS):
                            ts = random.randint(LASTFM_MIN_TS, LASTFM_MAX_TS)
                        item['timestamp'] = ts
                    except (ValueError, TypeError):
                        print(f"  无效的时间戳: {item['timestamp']}，使用默认值")
                        item['timestamp'] = random.randint(LASTFM_MIN_TS, LASTFM_MAX_TS)

                    # 验证其他字段
                    if not isinstance(item['artistID'], (int, str)) or not item['artistID']:
                        print(f"  无效的 artistID: {item['artistID']}，跳过")
                        continue
                    if not isinstance(item['tagValue'], str) or not item['tagValue']:
                        print(f"  无效的 tagValue: {item['tagValue']}，跳过")
                        continue

                    valid_data.append(item)

                user_results[bias_type] = valid_data
                print(f"  成功保存 {len(valid_data)} 条行为记录")
                total_success += 1

            except Exception as e:
                print(f"  处理失败: {str(e)[:100]}")
                import traceback
                traceback.print_exc()
                user_results[bias_type] = []
                total_failed += 1

            time.sleep(2)  # 避免速率限制

        results[user_id] = {
            "user_data": data['data'],
            "behavior_data": user_results
        }

        # 每完成一个用户保存一次进度
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"  用户 {user_id} 的进度已保存到 {output_file}")
        except Exception as e:
            print(f"  保存进度失败: {str(e)[:100]}")

    print(f"\n{'=' * 50}\n完成! 成功: {total_success}, 失败: {total_failed}")
    return results


def behaviors_to_dataframe(results):
    """
    将生成的行为数据转换为DataFrame（增强版）
    """
    all_behaviors = []

    for user_id, data in results.items():
        for bias_type, behaviors in data['behavior_data'].items():
            for behavior in behaviors:
                behavior.setdefault('userID', user_id)
                behavior.setdefault('bias_type', bias_type)
                for field in ['artistID', 'tagValue', 'weight', 'timestamp']:
                    behavior.setdefault(field, None)
                all_behaviors.append(behavior)

    df = pd.DataFrame(all_behaviors)

    # 转换数据类型
    if 'weight' in df.columns:
        df['weight'] = pd.to_numeric(df['weight'], errors='coerce')
        mask = df['weight'].isna() | (df['weight'] < 100)
        df.loc[mask, 'weight'] = df.loc[mask].apply(
            lambda _: random.randint(1000, 10000), axis=1
        )
        df['weight'] = df['weight'].astype(int)

    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
        mask = df['timestamp'].isna() | (df['timestamp'] < LASTFM_MIN_TS) | (df['timestamp'] > LASTFM_MAX_TS)
        df.loc[mask, 'timestamp'] = df.loc[mask].apply(
            lambda _: random.randint(LASTFM_MIN_TS, LASTFM_MAX_TS), axis=1
        )
        df['timestamp'] = df['timestamp'].astype(int)

    if 'artistID' in df.columns:
        df['artistID'] = df['artistID'].astype(str)  # 允许字符串或整数

    if 'tagValue' in df.columns:
        df['tagValue'] = df['tagValue'].astype(str)

    return df


# 主程序
# if __name__ == "__main__":
#     # 加载用户数据
#     user_df = combined_cold_start_data.sample(n = 200 , random_state=42)
#     # 生成行为数据
#     behavior_results = generate_behavior_data(user_df, "D:/user_behaviors.json")
#     # 转换为DataFrame
#     behavior_df = behaviors_to_dataframe(behavior_results)
#     print("最终行为数据：")
#     print(behavior_df.head())
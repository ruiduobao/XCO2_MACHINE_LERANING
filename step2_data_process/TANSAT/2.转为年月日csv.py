import pandas as pd
from pathlib import Path
import sys # To exit if input file not found

# --- 1. Configuration ---

# <<< 重要: 请将这里替换为你的实际输入 CSV 文件路径 >>>
INPUT_CSV_PATH = Path(r"E:\地理所\论文\中国XCO2论文_2025.04\数据\Tansat\原始数据_extracted_csv\TanSat_L2_XCO2_lite_2018_filtered_region.csv")

# <<< 重要: 请将这里替换为你想要保存的新 CSV 文件路径 >>>
OUTPUT_CSV_PATH = Path(r"E:\地理所\论文\中国XCO2论文_2025.04\数据\Tansat\原始数据_extracted_csv\TanSat_L2_XCO2_lite_all_extracted_time_parsed.csv")

# <<< 重要: 确认你的 CSV 文件中代表数值时间的列名 >>>
# 根据你给的示例数据行，列名是 'time'
# 如果你用了上一个脚本生成的，可能是 'time_unix'
TIME_COLUMN_NAME = 'time' # 或者 'time_unix'

# 检查输入文件是否存在
if not INPUT_CSV_PATH.is_file():
    print(f"错误：输入文件未找到: {INPUT_CSV_PATH}")
    sys.exit(1) # 退出脚本

# --- 2. 读取 CSV 文件 ---
print(f"正在读取输入 CSV 文件: {INPUT_CSV_PATH}")
try:
    df = pd.read_csv(INPUT_CSV_PATH)
    print(f"成功读取数据，原始形状: {df.shape}")
except Exception as e:
    print(f"错误: 读取 CSV 文件时出错: {e}")
    sys.exit(1) # 退出脚本

# --- 3. 时间转换与分列 ---
print(f"开始处理时间列 '{TIME_COLUMN_NAME}'...")

# 检查时间列是否存在
if TIME_COLUMN_NAME not in df.columns:
    print(f"错误: 在 CSV 文件中未找到指定的时间列 '{TIME_COLUMN_NAME}'。可用列为: {df.columns.tolist()}")
    sys.exit(1) # 退出脚本

try:
    # --- 时间转换 ---
    # 假设 TIME_COLUMN_NAME 是自 1970-01-01 00:00:00 UTC 以来的秒数 (Unix Epoch)
    # !! 如果纪元或单位不同，必须修改 unit/origin 或采用其他转换 !!
    print(f"将 '{TIME_COLUMN_NAME}' (假设为秒) 转换为 datetime 对象...")
    # errors='coerce' 会将无效转换变为 NaT (Not a Time)
    df['datetime'] = pd.to_datetime(df[TIME_COLUMN_NAME], unit='s', origin='unix', errors='coerce')

    # 检查转换后 NaT 的数量
    nat_count = df['datetime'].isnull().sum()
    if nat_count > 0:
        print(f"警告: 时间转换中有 {nat_count} 个值无法解析，已设为 NaT。")
        # 移除时间转换失败的行
        initial_rows = len(df)
        df.dropna(subset=['datetime'], inplace=True)
        print(f"已移除 {initial_rows - len(df)} 行时间无效的数据。")
        if df.empty:
            print("错误: 移除无效时间后 DataFrame 为空。")
            sys.exit(1) # 退出脚本

    # --- 提取时间分量 ---
    print("提取 年, 月, 日, 时, 分, 秒 ...")
    df['year'] = df['datetime'].dt.year
    df['month'] = df['datetime'].dt.month
    df['day'] = df['datetime'].dt.day
    df['hour'] = df['datetime'].dt.hour
    df['minute'] = df['datetime'].dt.minute
    df['second'] = df['datetime'].dt.second
    # df['microsecond'] = df['datetime'].dt.microsecond # 如果需要微秒

    # --- 清理列 ---
    # 删除临时的 datetime 列和原始的数值 time 列
    columns_to_drop = ['datetime', TIME_COLUMN_NAME]
    # 只删除存在的列，避免错误
    columns_to_drop = [col for col in columns_to_drop if col in df.columns]
    df = df.drop(columns=columns_to_drop)
    print("时间分列完成，并已删除原始时间列和临时 datetime 列。")

    # --- 调整列顺序 (可选) ---
    # 将新的时间列放到经纬度后面
    time_cols = ['year', 'month', 'day', 'hour', 'minute', 'second']
    # 获取其他所有列
    other_cols = [col for col in df.columns if col not in time_cols + ['sounding_id', 'latitude', 'longitude']]
    # 定义新顺序
    new_col_order = ['sounding_id', 'latitude', 'longitude'] + time_cols + other_cols
    # 应用新顺序 (只在所有列都存在时进行)
    if all(col in df.columns for col in new_col_order):
         df = df[new_col_order]
         print("已调整列顺序。")
    else:
         print("警告：未能调整列顺序，可能部分列名不存在。")


    # --- 4. 保存结果到新的 CSV 文件 ---
    print(f"最终 DataFrame 形状: {df.shape}")
    print(f"正在保存处理后的 DataFrame 到 CSV 文件: {OUTPUT_CSV_PATH}")
    try:
        df.to_csv(OUTPUT_CSV_PATH, index=False, encoding='utf-8-sig')
        print(f"数据成功保存至: {OUTPUT_CSV_PATH}")
    except Exception as e:
        print(f"错误: 保存 CSV 文件时出错: {e}")

except Exception as e:
    print(f"处理过程中发生意外错误: {e}")
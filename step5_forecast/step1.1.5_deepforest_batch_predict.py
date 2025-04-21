import pandas as pd
import pickle
import numpy as np
import os
import rasterio
import time
import glob
import re
from datetime import datetime

# --- 配置 ---
# 模型文件路径
MODEL_PATH = r"E:\地理所\论文\中国XCO2论文_2025.04\处理结果\模型数据\deepforest\deepforest_xco2_regression_model.pkl"

# 输入CSV文件的目录和模式
INPUT_DIR = r"E:\地理所\论文\中国XCO2论文_2025.04\数据\训练表格数据\标准栅格数据\标准栅格XY_添加tif"
INPUT_PATTERN = "统计_*.csv"  # 例如: 统计_2008_03.csv, 统计_2008_04.csv 等

# 输出目录
OUTPUT_DIR = r"E:\地理所\论文\中国XCO2论文_2025.04\数据\训练表格数据\标准栅格数据\标准栅格XY_添加tif_预测XCO2"

# 源栅格路径
SOURCE_RASTER_PATH = r"E:\地理所\论文\中国XCO2论文_2025.04\数据\范围数据\标准栅格_WGS84.tif"

# 需要从特征中排除的列名 (这些列不会用于预测，但会保留在最终结果中)
COLUMNS_TO_EXCLUDE = ['X', 'Y']

# 缺失值处理
MISSING_VALUE = -9999  # 缺失值标记
REPLACE_MISSING_WITH = 0  # 将缺失值替换为0

# 输出栅格参数
OUTPUT_DTYPE = 'float32'
OUTPUT_NODATA = -9999.0

# 年份和月份范围 (可选，用于过滤文件)
# 如果想处理所有文件，保持为None
START_YEAR = 2008
END_YEAR = 2023
# 如果只想处理特定月份，例如1月和7月，可以设置为 [1, 7]
# 如果想处理所有月份，保持为None
MONTHS_TO_PROCESS = None  # None表示所有月份, 或者指定月份列表，如 [1, 4, 7, 10]

# --- 实用函数 ---
def extract_year_month(filename):
    """从文件名中提取年份和月份"""
    # 适用于类似"统计_2008_03.csv"的文件名
    pattern = r'统计_(\d{4})_(\d{2})\.csv'
    match = re.search(pattern, filename)
    if match:
        year = int(match.group(1))
        month = int(match.group(2))
        return year, month
    return None, None

def make_predictions(model, input_file, output_file, cols_to_exclude, expected_features=None):
    """使用Deep Forest模型进行预测"""
    start_time = time.time()
    
    try:
        # 加载需要预测的数据
        prediction_data = pd.read_csv(input_file)
        print(f"待预测数据加载成功。 数据形状: {prediction_data.shape}")
        
        # 保存坐标列用于后续重新添加到结果中
        coordinates = {}
        for col in cols_to_exclude:
            if col in prediction_data.columns:
                coordinates[col] = prediction_data[col].copy()
        
        if coordinates:
            print(f"保存坐标列用于结果输出: {list(coordinates.keys())}")
        
        # 选择特征列（基于模型期望的特征名称）
        if expected_features:
            print(f"根据模型训练时的特征名称选择特征列...")
            
            # 检查是否有模型需要的特征在当前数据中缺失
            missing_cols = set(expected_features) - set(prediction_data.columns)
            if missing_cols:
                print(f"错误: 待预测数据中缺少模型需要的以下特征列: {missing_cols}")
                print("请确保预测数据包含所有模型需要的特征列。")
                return None
            
            # 使用模型需要的特征列
            X_predict = prediction_data[expected_features].copy()
            print(f"已选择 {len(expected_features)} 个特征列用于预测。")
        else:
            # 没有特征名称信息时，使用除排除列外的所有列
            print("无法获取模型特征名称，将使用除排除列外的所有列作为特征")
            X_predict = prediction_data.drop(columns=cols_to_exclude, errors='ignore').copy()
            print(f"已选择 {X_predict.shape[1]} 个特征列用于预测。")
        
        # 应用与训练时相同的预处理
        # 替换-9999
        missing_count = (X_predict == MISSING_VALUE).sum().sum()
        if missing_count > 0:
            print(f"检测到 {missing_count} 个值为 {MISSING_VALUE} 的缺失值标记，将替换为 {REPLACE_MISSING_WITH}")
            X_predict.replace(MISSING_VALUE, REPLACE_MISSING_WITH, inplace=True)
        
        # 替换无穷大值为NaN
        X_predict.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # 将所有列转换为数值类型
        X_predict = X_predict.apply(pd.to_numeric, errors='coerce')
        
        # 检查NaN数量
        nan_count = X_predict.isna().sum().sum()
        if nan_count > 0:
            print(f"检测到 {nan_count} 个NaN值，将替换为 {REPLACE_MISSING_WITH}")
            X_predict.fillna(REPLACE_MISSING_WITH, inplace=True)
        
        # 进行预测
        predicted_xco2 = model.predict(X_predict.values)
        print(f"预测完成。共生成 {len(predicted_xco2)} 个预测值。")
        
        # 创建一个新的DataFrame，包含坐标列和预测结果
        output_data = pd.DataFrame()
        
        # 添加坐标列
        for col, values in coordinates.items():
            output_data[col] = values
        
        # 添加预测列
        output_data['predicted_xco2'] = predicted_xco2
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # 保存到新的 CSV 文件
        output_data.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"预测结果已成功保存到: {output_file}")
        
        prediction_time = time.time() - start_time
        print(f"预测完成，耗时: {prediction_time:.2f} 秒")
        
        return output_data
    
    except Exception as e:
        print(f"预测过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return None

def create_tif_from_csv(csv_path, source_raster_path, output_tif_path):
    """将CSV中的预测结果赋值给TIF文件"""
    try:
        # 读取 CSV 数据
        df = pd.read_csv(csv_path)
        
        # 检查必需的列是否存在
        required_cols = ['X', 'Y', 'predicted_xco2']
        if not all(col in df.columns for col in required_cols):
            missing_cols = [col for col in required_cols if col not in df.columns]
            raise ValueError(f"CSV 文件缺少必需的列: {', '.join(missing_cols)}")
        
        # 打开源栅格
        with rasterio.open(source_raster_path) as src:
            profile = src.profile.copy()  # 获取源栅格的元数据
            source_data = src.read(1)     # 读取源栅格第一个波段的数据
            
            # 准备输出栅格的数据数组
            output_data = np.full(source_data.shape, OUTPUT_NODATA, dtype=OUTPUT_DTYPE)
            
            # 遍历 CSV 行，更新输出栅格数组
            updated_count = 0
            
            # 使用迭代器遍历 DataFrame 行
            for index, row in df.iterrows():
                x = int(row['X'])  # 列号
                y = int(row['Y'])  # 行号
                xco2_value = row['predicted_xco2']
                
                # 检查坐标是否在栅格范围内
                if 0 <= y < source_data.shape[0] and 0 <= x < source_data.shape[1]:
                    # 检查源栅格在该位置的值是否为 1
                    if source_data[y, x] == 1:
                        # 如果是 1，则将 CSV 中的 xco2 值赋给输出数组的对应位置
                        output_data[y, x] = xco2_value
                        updated_count += 1
            
            # 更新输出栅格的元数据
            profile.update(dtype=OUTPUT_DTYPE, nodata=OUTPUT_NODATA, count=1)
            
            # 确保输出目录存在
            os.makedirs(os.path.dirname(output_tif_path), exist_ok=True)
            
            # 写入新的栅格文件
            with rasterio.open(output_tif_path, 'w', **profile) as dst:
                dst.write(output_data, 1) # 将更新后的数据写入第一个波段
            
        print(f"  成功更新 {updated_count} 个像元，并保存到: {output_tif_path}")
        return True
    
    except Exception as e:
        print(f"创建TIF时发生错误: {e}")
        import traceback
        traceback.print_exc()
        return False

# --- 主脚本 ---
if __name__ == "__main__":
    print("=" * 80)
    print("开始 Deep Forest XCO2 批量预测")
    print("=" * 80)
    
    start_time = time.time()
    
    # 1. 加载模型
    print(f"[1/4] 正在加载模型: {MODEL_PATH}")
    if not os.path.exists(MODEL_PATH):
        print(f"错误: 模型文件未找到 {MODEL_PATH}")
        exit()
    
    try:
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        print("模型加载成功。")
        
        # 尝试提取特征名称
        # 尝试从同目录下的特征名称文件加载
        feature_names_path = os.path.splitext(MODEL_PATH)[0] + "_feature_names.txt"
        if os.path.exists(feature_names_path):
            with open(feature_names_path, 'r') as f:
                expected_features = [line.strip() for line in f.readlines()]
            print(f"从特征名称文件中加载了 {len(expected_features)} 个特征名。")
        else:
            print("警告：无法找到特征名称文件，也未找到特征名称文件。")
            expected_features = None
    except Exception as e:
        print(f"加载模型时发生错误: {e}")
        exit()
    
    # 2. 查找所有满足条件的输入文件
    print(f"[2/4] 查找输入文件: {os.path.join(INPUT_DIR, INPUT_PATTERN)}")
    all_files = glob.glob(os.path.join(INPUT_DIR, INPUT_PATTERN))
    print(f"找到 {len(all_files)} 个文件")
    
    # 过滤文件（根据年份和月份）
    filtered_files = []
    for file_path in all_files:
        filename = os.path.basename(file_path)
        year, month = extract_year_month(filename)
        
        if year is None or month is None:
            print(f"  警告: 无法从 {filename} 提取年份和月份，跳过此文件")
            continue
        
        # 检查年份范围
        if START_YEAR and year < START_YEAR:
            continue
        if END_YEAR and year > END_YEAR:
            continue
            
        # 检查月份
        if MONTHS_TO_PROCESS and month not in MONTHS_TO_PROCESS:
            continue
            
        filtered_files.append((file_path, year, month))
    
    # 按年份和月份排序文件
    filtered_files.sort(key=lambda x: (x[1], x[2]))
    
    print(f"筛选后剩余 {len(filtered_files)} 个文件待处理")
    if not filtered_files:
        print("没有符合条件的文件，退出程序")
        exit()
    
    # 3. 处理每个文件
    print(f"[3/4] 开始批量处理...")
    successful_files = 0
    failed_files = 0
    
    for i, (input_file, year, month) in enumerate(filtered_files):
        print(f"\n处理文件 {i+1}/{len(filtered_files)}: {os.path.basename(input_file)} (年份: {year}, 月份: {month})")
        
        # 构建输出文件路径
        filename = os.path.basename(input_file)
        output_csv = os.path.join(OUTPUT_DIR, filename.replace('.csv', '_deepforest_predictions.csv'))
        output_tif = os.path.join(OUTPUT_DIR, f"预测xco2_deepforest_{year}_{month:02d}.tif")
        
        # 进行预测
        print(f"  正在进行预测...")
        prediction_result = make_predictions(
            model, 
            input_file, 
            output_csv, 
            COLUMNS_TO_EXCLUDE,
            expected_features
        )
        
        # 如果预测成功，创建TIF
        if prediction_result is not None:
            print(f"  正在创建TIF文件...")
            tif_result = create_tif_from_csv(
                output_csv,
                SOURCE_RASTER_PATH,
                output_tif
            )
            
            if tif_result:
                successful_files += 1
                print(f"  文件 {os.path.basename(input_file)} 处理完成！")
            else:
                failed_files += 1
                print(f"  文件 {os.path.basename(input_file)} TIF创建失败。")
        else:
            failed_files += 1
            print(f"  文件 {os.path.basename(input_file)} 预测失败。")
    
    # 4. 总结
    total_time = time.time() - start_time
    print("\n" + "=" * 80)
    print("[4/4] 批量处理完成!")
    print(f"总共处理: {len(filtered_files)} 个文件")
    print(f"成功处理: {successful_files} 个文件")
    print(f"处理失败: {failed_files} 个文件")
    print(f"总耗时: {total_time:.2f} 秒, 平均每个文件: {total_time/len(filtered_files):.2f} 秒")
    print("=" * 80) 
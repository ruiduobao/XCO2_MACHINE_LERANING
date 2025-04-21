import pandas as pd
import joblib
import numpy as np
import os
import rasterio
import time

# --- 配置 ---
# 模型和数据文件路径
MODEL_PATH = r"E:\地理所\论文\中国XCO2论文_2025.04\处理结果\模型数据\lightGBM\lightgbm_xco2_regression_model.pkl"
PREDICTION_INPUT_FILE = r"E:\地理所\论文\中国XCO2论文_2025.04\数据\训练表格数据\标准栅格数据\标准栅格XY_添加tif\统计_2008_03.csv"
PREDICTION_OUTPUT_FILE = r"E:\地理所\论文\中国XCO2论文_2025.04\数据\训练表格数据\标准栅格数据\标准栅格XY_添加tif\统计_2008_03_lightgbm_predictions.csv"

# 源栅格和输出栅格路径
SOURCE_RASTER_PATH = r"E:\地理所\论文\中国XCO2论文_2025.04\数据\范围数据\标准栅格_WGS84.tif"
OUTPUT_RASTER_PATH = r"E:\地理所\论文\中国XCO2论文_2025.04\数据\训练表格数据\标准栅格数据\标准栅格XY_添加tif_预测XCO2\预测xco2_lightgbm.tif"

# 需要从特征中排除的列名 (这些列不会用于预测，但会保留在最终结果中)
COLUMNS_TO_EXCLUDE = ['X', 'Y']

# 输出栅格参数
OUTPUT_DTYPE = 'float32'
OUTPUT_NODATA = -9999.0

# --- 函数定义 ---
def make_predictions(model_path, input_file, output_file, cols_to_exclude):
    """使用LightGBM模型进行预测"""
    print(f"=== 步骤1：使用 LightGBM 模型进行预测 ===")
    start_time = time.time()
    
    # 1. 加载已训练的模型
    print(f"\n[1/5] 正在加载模型从: {model_path}")
    if not os.path.exists(model_path):
        print(f"错误: 模型文件未找到 {model_path}")
        return None
    
    try:
        model = joblib.load(model_path)
        print("模型加载成功。")
        
        # 尝试提取特征名称
        if hasattr(model, 'feature_name_'):
            expected_features = model.feature_name_
        else:
            # 尝试从同目录下的特征名称文件加载
            feature_names_path = os.path.splitext(model_path)[0] + "_feature_names.txt"
            if os.path.exists(feature_names_path):
                with open(feature_names_path, 'r') as f:
                    expected_features = [line.strip() for line in f.readlines()]
                print(f"从特征名称文件中加载了 {len(expected_features)} 个特征名。")
            else:
                print("警告：无法从模型中提取特征名称，也未找到特征名称文件。")
                expected_features = None
                
        if expected_features:
            print(f"模型训练时使用了 {len(expected_features)} 个特征。")
    except Exception as e:
        print(f"加载模型时发生错误: {e}")
        return None

    # 2. 加载需要预测的数据
    print(f"\n[2/5] 正在加载待预测数据从: {input_file}")
    if not os.path.exists(input_file):
        print(f"错误: 待预测文件未找到 {input_file}")
        return None
    
    try:
        prediction_data = pd.read_csv(input_file)
        print(f"待预测数据加载成功。 数据形状: {prediction_data.shape}")
    except Exception as e:
        print(f"加载待预测数据时出错: {e}")
        return None

    # 3. 准备特征矩阵 (X_predict)
    print("\n[3/5] 准备用于预测的特征矩阵...")

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
    print("将特征数据转换为数值类型并处理 Inf 值...")
    X_predict.replace([np.inf, -np.inf], np.nan, inplace=True)
    X_predict = X_predict.apply(pd.to_numeric, errors='coerce')

    # 检查 NaN 情况
    nan_counts_predict = X_predict.isna().sum().sum()
    if nan_counts_predict > 0:
        print(f"注意：预测数据中存在 {nan_counts_predict} 个 NaN 值。LightGBM 将根据训练时的策略处理它们。")

    print(f"准备好的预测特征矩阵形状: {X_predict.shape}")

    # 4. 进行预测
    print("\n[4/5] 使用加载的模型进行预测...")
    try:
        predicted_xco2 = model.predict(X_predict)
        print(f"预测完成。共生成 {len(predicted_xco2)} 个预测值。")
    except Exception as e:
        print(f"预测过程中发生错误: {e}")
        return None

    # 5. 保存结果
    print("\n[5/5] 将预测结果添加到原始数据并保存...")
    
    # 创建一个新的DataFrame，包含坐标列和预测结果
    output_data = pd.DataFrame()
    
    # 添加坐标列
    for col, values in coordinates.items():
        output_data[col] = values
    
    # 添加预测列
    output_data['predicted_xco2'] = predicted_xco2

    try:
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # 保存到新的 CSV 文件
        output_data.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"预测结果已成功保存到: {output_file}")
    except Exception as e:
        print(f"保存预测结果时出错: {e}")
        return None
        
    prediction_time = time.time() - start_time
    print(f"\n预测步骤完成，耗时: {prediction_time:.2f} 秒")
    
    return output_data

def assign_predictions_to_tif(csv_path, source_raster_path, output_raster_path, output_dtype, output_nodata):
    """将CSV中的预测结果赋值给TIF文件"""
    print(f"\n=== 步骤2：将预测结果赋值给TIF文件 ===")
    start_time = time.time()
    
    try:
        # --- 1. 读取 CSV 数据 ---
        print(f"正在读取 CSV 文件: {csv_path} ...")
        df = pd.read_csv(csv_path)
        print(f"成功读取 {len(df)} 行数据。")

        # 检查必需的列是否存在
        required_cols = ['X', 'Y', 'predicted_xco2']
        if not all(col in df.columns for col in required_cols):
            missing_cols = [col for col in required_cols if col not in df.columns]
            raise ValueError(f"CSV 文件缺少必需的列: {', '.join(missing_cols)}")
        print("CSV 列检查通过。")

        # --- 2. 打开源栅格 ---
        print(f"正在打开源栅格文件: {source_raster_path} ...")
        with rasterio.open(source_raster_path) as src:
            profile = src.profile.copy()  # 获取源栅格的元数据
            source_data = src.read(1)     # 读取源栅格第一个波段的数据
            print(f"源栅格维度: {source_data.shape} (行, 列)")
            print("源栅格元数据和数据读取成功。")

            # --- 3. 准备输出栅格的数据数组 ---
            print(f"正在创建输出栅格的数据数组，使用 NoData 值: {output_nodata} ...")
            # 创建一个与源栅格形状相同的数组，用 NoData 值填充
            output_data = np.full(source_data.shape, output_nodata, dtype=output_dtype)

            # --- 4. 遍历 CSV 行，更新输出栅格数组 ---
            print("正在根据 CSV 数据更新输出栅格的值...")
            updated_count = 0
            skipped_source_not_1 = 0
            skipped_out_of_bounds = 0

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
                    else:
                        # 如果源栅格值不是 1，则跳过
                        skipped_source_not_1 += 1
                else:
                    # 如果坐标超出范围，则跳过
                    skipped_out_of_bounds += 1

                # 进度指示器
                if (index + 1) % 10000 == 0:
                    print(f"  已处理 {index + 1}/{len(df)} 行...")

            print("栅格值更新完成。")
            print(f"  成功更新 {updated_count} 个像元。")
            if skipped_source_not_1 > 0:
                print(f"  跳过 {skipped_source_not_1} 个像元 (因为源栅格对应位置值不为 1)。")
            if skipped_out_of_bounds > 0:
                print(f"  跳过 {skipped_out_of_bounds} 个像元 (因为 CSV 中的 X, Y 坐标超出栅格范围)。")

            # --- 5. 更新输出栅格的元数据 ---
            print("正在更新输出栅格的元数据...")
            profile.update(dtype=output_dtype, nodata=output_nodata, count=1)
            print(f"  数据类型设置为: {output_dtype}")
            print(f"  NoData 值设置为: {output_nodata}")

            # --- 6. 写入新的栅格文件 ---
            # 确保输出目录存在
            os.makedirs(os.path.dirname(output_raster_path), exist_ok=True)
            
            print(f"正在写入新的栅格文件: {output_raster_path} ...")
            with rasterio.open(output_raster_path, 'w', **profile) as dst:
                dst.write(output_data, 1) # 将更新后的数据写入第一个波段
            print("新的栅格文件写入成功！")
            
        tif_time = time.time() - start_time
        print(f"\nTIF生成步骤完成，耗时: {tif_time:.2f} 秒")
        return True

    except FileNotFoundError as e:
        print(f"错误：文件未找到 - 请检查路径 '{e.filename}' 是否正确。")
    except ValueError as e:
        print(f"错误：数据或配置问题 - {e}")
    except KeyError as e:
        print(f"错误: 在 CSV 文件中找不到列 '{e}'。")
    except Exception as e:
        print(f"处理过程中发生未知错误: {e}")
        import traceback
        traceback.print_exc()
    
    return False

# --- 主脚本 ---
if __name__ == "__main__":
    print("=" * 80)
    print("开始 LightGBM XCO2 预测工作流")
    print("=" * 80)
    
    total_start_time = time.time()
    
    # 步骤1: 使用LightGBM模型进行预测
    prediction_result = make_predictions(
        MODEL_PATH, 
        PREDICTION_INPUT_FILE, 
        PREDICTION_OUTPUT_FILE, 
        COLUMNS_TO_EXCLUDE
    )
    
    # 如果预测成功，继续步骤2
    if prediction_result is not None:
        # 步骤2: 将预测值赋值给TIF文件
        tif_result = assign_predictions_to_tif(
            PREDICTION_OUTPUT_FILE,
            SOURCE_RASTER_PATH,
            OUTPUT_RASTER_PATH,
            OUTPUT_DTYPE,
            OUTPUT_NODATA
        )
        
        if tif_result:
            print("\n所有步骤已成功完成！")
        else:
            print("\n步骤2(TIF生成)失败。")
    else:
        print("\n步骤1(预测)失败，流程中止。")
    
    total_time = time.time() - total_start_time
    print(f"\n整个工作流完成，总耗时: {total_time:.2f} 秒") 
import pandas as pd
import numpy as np
import os
import pickle
import time

# --- 配置 ---
MODEL_PATH = r"E:\地理所\论文\中国XCO2论文_2025.04\处理结果\模型数据\deepforest\deepforest_xco2_regression_model.pkl"  # 模型文件路径
PREDICTION_INPUT_FILE = r"E:\地理所\论文\中国XCO2论文_2025.04\数据\训练表格数据\标准栅格数据\标准栅格XY_添加tif\统计_2008_03.csv"  # 要预测的表格路径
PREDICTION_OUTPUT_FILE = r"E:\地理所\论文\中国XCO2论文_2025.04\数据\训练表格数据\标准栅格数据\标准栅格XY_添加tif\统计_2008_03_deepforest_predictions.csv"  # 输出文件路径和名称

# 需要从特征中排除的列名
COLUMNS_TO_EXCLUDE = ['X', 'Y']

# 缺失值处理
MISSING_VALUE = -9999  # 缺失值标记
REPLACE_MISSING_WITH = 0  # 将缺失值替换为0

# --- 主脚本 ---
if __name__ == "__main__":
    print(f"--- 开始使用 Deep Forest 模型进行预测 ---")

    # 1. 加载已训练的模型
    print(f"\n[1/5] 正在加载模型从: {MODEL_PATH}")
    if not os.path.exists(MODEL_PATH):
        print(f"错误: 模型文件未找到 {MODEL_PATH}")
        exit()
    try:
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        print("模型加载成功。")
        
        # 尝试从同目录下的特征名称文件加载特征名称
        feature_names_path = os.path.splitext(MODEL_PATH)[0] + "_feature_names.txt"
        if os.path.exists(feature_names_path):
            with open(feature_names_path, 'r') as f:
                expected_features = [line.strip() for line in f.readlines()]
            print(f"从特征名称文件中加载了 {len(expected_features)} 个特征名。")
        else:
            print("警告：无法找到特征名称文件。将尝试使用预测数据中的所有列（除排除列外）作为特征。")
            expected_features = None

    except Exception as e:
        print(f"加载模型时发生错误: {e}")
        exit()

    # 2. 加载需要预测的数据
    print(f"\n[2/5] 正在加载待预测数据从: {PREDICTION_INPUT_FILE}")
    if not os.path.exists(PREDICTION_INPUT_FILE):
        print(f"错误: 待预测文件未找到 {PREDICTION_INPUT_FILE}")
        exit()
    try:
        prediction_data = pd.read_csv(PREDICTION_INPUT_FILE)
        print(f"待预测数据加载成功。 数据形状: {prediction_data.shape}")
    except Exception as e:
        print(f"加载待预测数据时出错: {e}")
        exit()

    # 3. 准备特征矩阵 (X_predict)
    print("\n[3/5] 准备用于预测的特征矩阵...")

    # 保存坐标列用于后续重新添加到结果中
    coordinates = {}
    for col in COLUMNS_TO_EXCLUDE:
        if col in prediction_data.columns:
            coordinates[col] = prediction_data[col].copy()
    
    if coordinates:
        print(f"保存坐标列用于结果输出: {list(coordinates.keys())}")

    # 选择特征列
    if expected_features:
        print(f"根据模型训练时的特征名称选择特征列...")
        
        # 检查是否有模型需要的特征在当前数据中缺失
        missing_cols = set(expected_features) - set(prediction_data.columns)
        if missing_cols:
            print(f"错误: 待预测数据中缺少模型需要的以下特征列: {missing_cols}")
            print("请确保预测数据包含所有模型需要的特征列。")
            exit()
        
        # 使用模型需要的特征列
        X_predict = prediction_data[expected_features].copy()
        print(f"已选择 {len(expected_features)} 个特征列用于预测。")
    else:
        # 没有特征名称信息时，使用除排除列外的所有列
        print("无法获取模型特征名称，将使用除排除列外的所有列作为特征")
        X_predict = prediction_data.drop(columns=COLUMNS_TO_EXCLUDE, errors='ignore').copy()
        print(f"已选择 {X_predict.shape[1]} 个特征列用于预测。")

    # 应用与训练时相同的预处理
    print("将特征数据转换为数值类型并处理缺失值...")
    
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
    
    print(f"准备好的预测特征矩阵形状: {X_predict.shape}")

    # 4. 进行预测
    print("\n[4/5] 使用加载的模型进行预测...")
    try:
        start_time = time.time()
        predicted_xco2 = model.predict(X_predict.values)
        prediction_time = time.time() - start_time
        print(f"预测完成。共生成 {len(predicted_xco2)} 个预测值。")
        print(f"预测 {len(X_predict)} 个样本耗时: {prediction_time:.4f} 秒 "
              f"({len(X_predict)/prediction_time:.1f} 样本/秒)")
    except Exception as e:
        print(f"预测过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        exit()

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
        os.makedirs(os.path.dirname(PREDICTION_OUTPUT_FILE), exist_ok=True)
        
        # 保存到新的 CSV 文件
        output_data.to_csv(PREDICTION_OUTPUT_FILE, index=False, encoding='utf-8-sig')
        print(f"预测结果已成功保存到: {PREDICTION_OUTPUT_FILE}")
    except Exception as e:
        print(f"保存预测结果时出错: {e}")

    print("\n--- 预测脚本执行完毕 ---") 
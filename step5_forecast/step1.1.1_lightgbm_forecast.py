import pandas as pd
import joblib
import numpy as np
import os

# --- 配置 ---
MODEL_PATH = r"E:\地理所\论文\中国XCO2论文_2025.04\处理结果\模型数据\lightGBM\lightgbm_xco2_regression_model.pkl" # *修改为你实际的模型文件路径*
PREDICTION_INPUT_FILE = r"E:\地理所\论文\中国XCO2论文_2025.04\数据\训练表格数据\标准栅格数据\标准栅格XY_添加tif\统计_2008_03.csv" # *修改为你要预测的表格路径*
PREDICTION_OUTPUT_FILE = r"E:\地理所\论文\中国XCO2论文_2025.04\数据\训练表格数据\标准栅格数据\标准栅格XY_添加tif\统计_2008_03_lightgbm_predictions.csv" # *定义输出文件的路径和名称*

# 需要从特征中排除的列名 (这些列不会用于预测，但会保留在最终结果中)
COLUMNS_TO_EXCLUDE = ['X', 'Y']

# --- 主脚本 ---
if __name__ == "__main__":
    print(f"--- 开始使用 LightGBM 模型进行预测 ---")

    # 1. 加载已训练的模型
    print(f"\n[1/5] 正在加载模型从: {MODEL_PATH}")
    if not os.path.exists(MODEL_PATH):
        print(f"错误: 模型文件未找到 {MODEL_PATH}")
        exit()
    try:
        model = joblib.load(MODEL_PATH)
        print("模型加载成功。")
        
        # 尝试提取特征名称
        if hasattr(model, 'feature_name_'):
            expected_features = model.feature_name_
        else:
            # 尝试从同目录下的特征名称文件加载
            feature_names_path = os.path.splitext(MODEL_PATH)[0] + "_feature_names.txt"
            if os.path.exists(feature_names_path):
                with open(feature_names_path, 'r') as f:
                    expected_features = [line.strip() for line in f.readlines()]
                print(f"从特征名称文件中加载了 {len(expected_features)} 个特征名。")
            else:
                print("警告：无法从模型中提取特征名称，也未找到特征名称文件。")
                expected_features = None
                
        if expected_features:
            print(f"模型训练时使用了 {len(expected_features)} 个特征。")
            # print(f"预期的特征名称 (部分): {expected_features[:10]}...") # 打印前10个看看

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

    # 选择特征列（基于模型期望的特征名称）
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
        print("请检查特征矩阵的数据类型和值是否符合模型预期。")
        # 可以取消注释下面这行来查看数据信息以帮助调试
        # print(X_predict.info())
        # print(X_predict.head())
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
    
    # 还可以选择性地添加其他原始数据列
    # output_data = pd.concat([coordinates_df, output_data], axis=1)

    try:
        # 保存到新的 CSV 文件
        output_data.to_csv(PREDICTION_OUTPUT_FILE, index=False, encoding='utf-8-sig')
        print(f"预测结果已成功保存到: {PREDICTION_OUTPUT_FILE}")
    except Exception as e:
        print(f"保存预测结果时出错: {e}")

    print("\n--- 预测脚本执行完毕 ---") 
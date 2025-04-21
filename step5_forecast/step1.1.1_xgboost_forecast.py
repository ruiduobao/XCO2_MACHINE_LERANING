import pandas as pd
import xgboost as xgb
import numpy as np
import os

# --- 配置 ---
MODEL_PATH = r"F:\xgboost_xco2_regression_model.json" # *修改为你实际的模型文件路径*
PREDICTION_INPUT_FILE = r"E:\地理所\论文\中国XCO2论文_2025.04\数据\训练表格数据\标准栅格数据\标准栅格XY_添加tif\统计_2008_03.csv" # *修改为你要预测的表格路径*
PREDICTION_OUTPUT_FILE = r"E:\地理所\论文\中国XCO2论文_2025.04\数据\训练表格数据\标准栅格数据\标准栅格XY_添加tif\统计_2008_03_predictions.csv" # *定义输出文件的路径和名称*

# 需要从特征中排除的列名
COLUMNS_TO_EXCLUDE = ['X', 'Y']

# --- 主脚本 ---
if __name__ == "__main__":
    print(f"--- 开始使用 XGBoost 模型进行预测 ---")
    print(f"XGBoost 版本: {xgb.__version__}")

    # 1. 加载已训练的模型
    print(f"\n[1/5] 正在加载模型从: {MODEL_PATH}")
    if not os.path.exists(MODEL_PATH):
        print(f"错误: 模型文件未找到 {MODEL_PATH}")
        exit()
    try:
        model = xgb.XGBRegressor() # 初始化一个空的模型对象
        model.load_model(MODEL_PATH) # 加载训练好的模型参数
        print("模型加载成功。")
        # 提取模型训练时使用的特征名称 (非常重要)
        expected_features = model.get_booster().feature_names
        if not expected_features:
             print("警告：无法从模型中提取特征名称。请确保模型已正确保存。")
             # 在这种情况下，你可能需要手动提供特征列表
             # expected_features = ['year', 'month', 'Lantitude_band1', ...] # 手动列出所有特征
             # if not expected_features: # 如果还是没有，就退出
             #    print("错误：需要模型训练时的特征名称列表才能继续。")
             #    exit()
        else:
             print(f"模型训练时使用了 {len(expected_features)} 个特征。")
             # print(f"预期的特征名称 (部分): {expected_features[:10]}...") # 打印前10个看看

    except xgb.core.XGBoostError as e:
        print(f"错误: 加载模型失败。请检查路径或文件是否为有效的 XGBoost 模型。 XGBoost 错误: {e}")
        exit()
    except Exception as e:
        print(f"加载模型时发生未知错误: {e}")
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

    # 检查排除列是否存在
    actual_cols_to_exclude = [col for col in COLUMNS_TO_EXCLUDE if col in prediction_data.columns]
    if len(actual_cols_to_exclude) != len(COLUMNS_TO_EXCLUDE):
        missing_exclude = set(COLUMNS_TO_EXCLUDE) - set(actual_cols_to_exclude)
        print(f"警告：指定的排除列 {missing_exclude} 不在数据中。")

    # 选择特征列（所有列，除了要排除的）
    print(f"将从特征中排除列: {actual_cols_to_exclude}")
    potential_feature_cols = [col for col in prediction_data.columns if col not in actual_cols_to_exclude]
    if not potential_feature_cols:
        print("错误：排除指定列后没有剩余的特征列。")
        exit()

    X_predict = prediction_data[potential_feature_cols].copy() # 使用 .copy() 避免警告

    # --- 关键步骤: 确保特征与模型训练时一致 ---
    print(f"检查并对齐特征列与模型训练时的特征 ({len(expected_features)} 列)...")

    # 检查是否有模型需要的特征在当前数据中缺失
    missing_cols = set(expected_features) - set(X_predict.columns)
    if missing_cols:
        print(f"错误: 待预测数据中缺少模型需要的以下特征列: {missing_cols}")
        print("请确保预测数据的列名与训练数据（除目标变量和排除列外）一致。")
        exit()

    # 检查是否有模型训练时未使用的额外特征（这些将被忽略）
    extra_cols = set(X_predict.columns) - set(expected_features)
    if extra_cols:
        print(f"警告: 待预测数据中包含模型训练时未使用的额外列，这些列将被忽略: {extra_cols}")

    # 按模型训练时的顺序重新排序列，并只保留这些列
    try:
        X_predict = X_predict[expected_features]
        print(f"特征列已成功对齐为 {X_predict.shape[1]} 列。")
    except KeyError as e:
         print(f"错误: 尝试按模型预期顺序选择特征时出错。很可能是因为加载的特征列表 `expected_features` 与 `X_predict` 中的列不匹配。 错误: {e}")
         exit()
    except Exception as e:
         print(f"对齐特征列时发生未知错误: {e}")
         exit()


    # 应用与训练时相同的预处理
    print("将特征数据转换为数值类型并处理 Inf 值...")
    X_predict.replace([np.inf, -np.inf], np.nan, inplace=True)
    X_predict = X_predict.apply(pd.to_numeric, errors='coerce')

    # 检查 NaN 情况
    nan_counts_predict = X_predict.isna().sum().sum()
    if nan_counts_predict > 0:
        print(f"注意：预测数据中存在 {nan_counts_predict} 个 NaN 值。XGBoost 将根据训练时的策略处理它们。")

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
    # 创建原始数据的副本以添加新列
    output_data = prediction_data.copy()
    output_data['predicted_xco2'] = predicted_xco2 # 添加预测列

    try:
        # 保存到新的 CSV 文件
        output_data.to_csv(PREDICTION_OUTPUT_FILE, index=False, encoding='utf-8-sig')
        print(f"预测结果已成功保存到: {PREDICTION_OUTPUT_FILE}")
    except Exception as e:
        print(f"保存预测结果时出错: {e}")

    print("\n--- 预测脚本执行完毕 ---")
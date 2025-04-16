import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np
import subprocess # 用于调用 nvidia-smi 检查 GPU
import os

# --- 配置 ---
FILE_PATH = r"E:\地理所\论文\中国XCO2论文_2025.04\数据\训练表格数据\卫星数据\处理后的网格XCO2加权统计_优化版.csv"
TARGET_VARIABLE = 'xco2' # 你的目标变量列名
MODEL_SAVE_PATH = "xgboost_xco2_regression_model.json" # 模型保存路径和名称
TEST_SIZE = 0.2 # 测试集占总数据的比例
RANDOM_STATE = 42 # 随机种子，确保结果可复现

# --- 检查 GPU 可用性 ---
def check_gpu_availability():
    """检查系统中是否有可用的 NVIDIA GPU"""
    try:
        # 调用 nvidia-smi 检查 GPU
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, check=True)
        print("GPU 可用性检查：检测到 NVIDIA GPU。")
        # 你可以在这里添加更详细的解析来确认 CUDA 版本等
        return True
    except FileNotFoundError:
        print("GPU 可用性检查：'nvidia-smi' 未找到。请确保已安装 NVIDIA 驱动。将使用 CPU。")
        return False
    except subprocess.CalledProcessError:
        print("GPU 可用性检查：'nvidia-smi' 执行出错。可能没有可用的 NVIDIA GPU。将使用 CPU。")
        return False
    except Exception as e:
        print(f"GPU 可用性检查：发生未知错误: {e}。将使用 CPU。")
        return False

# --- 主脚本 ---
if __name__ == "__main__":
    print(f"--- 开始 XGBoost 回归模型训练 ---")
    print(f"XGBoost 版本: {xgb.__version__}")

    # 检查 GPU
    use_gpu = check_gpu_availability()
    device = 'cuda' if use_gpu else 'cpu'
    print(f"将使用 '{device}' 进行训练。")

    # 1. 加载数据
    try:
        print(f"\n[1/7] 正在加载数据从: {FILE_PATH}")
        data = pd.read_csv(FILE_PATH)
        print(f"数据加载成功。 数据形状: {data.shape}")
        # print(f"数据列名: {data.columns.tolist()}") # 如果列很多，可以注释掉这行
    except FileNotFoundError:
        print(f"错误: 文件未找到 {FILE_PATH}")
        exit()
    except Exception as e:
        print(f"加载数据时出错: {e}")
        exit()

    # 2. 数据预处理：识别目标和特征
    print(f"\n[2/7] 正在识别目标变量和特征变量...")
    if TARGET_VARIABLE not in data.columns:
        print(f"错误: 目标变量 '{TARGET_VARIABLE}' 不在数据列中。可用列: {data.columns.tolist()}")
        exit()

    # 替换无穷大值（如果存在）
    data.replace([np.inf, -np.inf], np.nan, inplace=True)

    try:
        target_column_index = data.columns.get_loc(TARGET_VARIABLE)
        y = data[TARGET_VARIABLE]
        X = data.iloc[:, target_column_index + 1:] # 选择目标变量之后的所有列作为特征

        print(f"目标变量 '{TARGET_VARIABLE}' 已找到。")
        print(f"使用其后的 {X.shape[1]} 列作为特征。")
        # print(f"特征列: {X.columns.tolist()}") # 如果列很多，可以注释掉

        # 确保所有特征列都是数值类型，非数值转为NaN
        print("将特征列转换为数值类型（非数值将变为NaN）...")
        X = X.apply(pd.to_numeric, errors='coerce')

        # 检查并移除完全是 NaN 的列（如果因转换产生）
        all_nan_cols = X.columns[X.isna().all()].tolist()
        if all_nan_cols:
            print(f"警告: 以下特征列在转换后全为NaN，将被移除: {all_nan_cols}")
            X = X.drop(columns=all_nan_cols)
            print(f"剩余特征数量: {X.shape[1]}")

        # 检查目标变量是否有 NaN 或 Inf
        if y.isna().any() or not np.isfinite(y).all():
            print(f"警告: 目标变量 '{TARGET_VARIABLE}' 包含 NaN 或 Inf 值。可能需要处理这些值。")
            print(f"NaN 数量: {y.isna().sum()}")
            # 可以选择填充或删除这些行，例如：
            # valid_indices = y.notna() & np.isfinite(y)
            # X = X[valid_indices]
            # y = y[valid_indices]
            # print(f"处理后数据形状: X={X.shape}, y={y.shape}")


        # 检查特征中的 NaN 值
        nan_counts = X.isna().sum()
        print("\n特征中的NaN值统计 (只显示包含NaN的列):")
        print(nan_counts[nan_counts > 0])
        if nan_counts.sum() > 0:
             print("注意：XGBoost 可以自动处理 NaN 值。")
        else:
             print("特征中没有检测到 NaN 值。")

    except Exception as e:
        print(f"处理目标变量和特征时出错: {e}")
        exit()


    # 3. 数据分割
    print(f"\n[3/7] 正在将数据拆分为训练集和测试集 (测试集比例: {TEST_SIZE})...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f"训练集大小: X={X_train.shape}, y={y_train.shape}")
    print(f"测试集大小: X={X_test.shape}, y={y_test.shape}")

    # 4. 初始化 XGBoost 回归模型
    print("\n[4/7] 初始化 XGBoost 回归模型...")
    model = xgb.XGBRegressor(
        objective='reg:squarederror', # 回归任务，最小化平方误差
        n_estimators=200,             # 树的数量 (可调优)
        learning_rate=0.1,            # 学习率 (可调优)
        max_depth=7,                  # 树的最大深度 (可调优)
        subsample=0.8,                # 每棵树使用的样本比例 (可调优)
        colsample_bytree=0.8,         # 每棵树使用的特征比例 (可调优)
        random_state=RANDOM_STATE,    # 随机种子
        tree_method='hist',           # 使用 histogram 算法 (通常更快)
        device=device,                # 设置为 'cuda' 或 'cpu'
        early_stopping_rounds=10,     # 提前停止轮数
        eval_metric='rmse'            # 使用 RMSE 作为早停的评估指标
    )
    print("模型参数设置完成。")

    # 5. 训练模型
    print("\n[5/7] 开始训练模型...")
    print("注意：训练过程可能会输出每次迭代的信息（如果 verbose=True）。")
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],  # 使用测试集进行评估和早停
        verbose=False # 设置为 True 可以看到详细的训练过程和每一轮的RMSE
        # verbose=50 # 每50轮打印一次信息
    )
    print("模型训练完成。")
    # 显示最佳迭代次数（如果使用了早停）
    if model.best_iteration is not None:
         print(f"最佳迭代次数（基于早停）: {model.best_iteration}")


    # 6. 在测试集上进行预测与评估
    print("\n[6/7] 在测试集上进行预测并评估模型性能...")
    y_pred = model.predict(X_test)

    # 计算评估指标
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\n--- 模型评估结果 (测试集) ---")
    print(f"均方根误差 (RMSE): {rmse:.4f}")
    print(f"平均绝对误差 (MAE): {mae:.4f}")
    print(f"R² 分数 (决定系数): {r2:.4f}")
    print("---------------------------------")
    print("说明:")
    print("  - RMSE/MAE: 预测值与真实值之间的平均误差（单位与xco2相同）。越低越好。")
    print("  - R² 分数: 模型解释的方差比例。范围从 -∞ 到 1，越接近 1 越好。")

    # 关于"评价经度"
    print("\n关于'评价经度':")
    print("  - 'longitude' (经度) 是用于预测 'xco2' 的 *特征* 之一。")
    print("  - 以上评估指标 (RMSE, MAE, R²) 是衡量整个模型预测 *'xco2'* 准确性的。")
    print("  - 如果想了解 'longitude' 等特征对预测的贡献大小，可以查看下面的'特征重要性'。")

    # 显示特征重要性
    try:
        print("\n特征重要性 (Top 15):")
        # XGBoost >= 1.6.0 推荐使用 'weight', 'gain', 'cover', 'total_gain', 'total_cover'
        # 默认为 'weight' (特征在所有树中被用于分裂的次数)
        # 'gain' (使用该特征进行分裂带来的平均增益) 通常更有信息量
        feature_importances = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_ # 默认是 'weight'
        }).sort_values('importance', ascending=False)

        # # 如果想看 'gain' 类型的重要性 (可能需要模型重新训练或用不同方式获取)
        # try:
        #     gain_importances = model.get_booster().get_score(importance_type='gain')
        #     feature_importances_gain = pd.DataFrame({
        #         'feature': gain_importances.keys(),
        #         'importance_gain': gain_importances.values()
        #     }).sort_values('importance_gain', ascending=False)
        #     print("\n特征重要性 (Gain, Top 15):")
        #     print(feature_importances_gain.head(15))
        # except Exception as e_gain:
        #      print(f"无法获取 'gain' 类型的重要性: {e_gain}")
        #      print("显示默认类型的重要性 (通常是 'weight'):")
        print(feature_importances.head(15))

    except Exception as e:
        print(f"无法获取或显示特征重要性: {e}")


    # 7. 保存模型
    print(f"\n[7/7] 正在保存训练好的模型到: {MODEL_SAVE_PATH}")
    try:
        model.save_model(MODEL_SAVE_PATH)
        print("模型已成功保存。")
    except Exception as e:
        print(f"保存模型时出错: {e}")

    print("\n--- 脚本执行完毕 ---")
import pandas as pd
from deepforest import CascadeForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np
import pickle
import os
import time
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 非交互式后端，适用于无显示器环境

# --- 配置 ---
FILE_PATH = r"E:\地理所\论文\中国XCO2论文_2025.04\数据\训练表格数据\卫星数据\处理后的网格XCO2加权统计_优化版.csv"
TARGET_VARIABLE = 'xco2'  # 目标变量列名
MODEL_SAVE_PATH = r"E:\地理所\论文\中国XCO2论文_2025.04\处理结果\模型数据\deepforest\deepforest_xco2_regression_model.pkl"  # 模型保存路径
TEST_SIZE = 0.2  # 测试集占总数据的比例
RANDOM_STATE = 42  # 随机种子，确保结果可复现
MISSING_VALUE = -9999  # 缺失值标记
REPLACE_MISSING_WITH = 0  # 将缺失值替换为0

# --- 数据预处理函数 ---
def preprocess_data(data, missing_value=-9999, replace_with=0):
    """
    预处理数据，处理缺失值和异常值
    
    Args:
        data (DataFrame): 输入数据
        missing_value (float): 需要替换的缺失值标记 (如-9999)
        replace_with (float): 替换缺失值的值 (如0)
        
    Returns:
        DataFrame, Series: 处理后的特征(X)和目标变量(y)
    """
    print(f"\n[2.1/7] 处理缺失值和异常值...")
    
    # 创建数据副本避免修改原始数据
    df = data.copy()
    
    # 替换极端值/无穷大值为NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # 查找并替换-9999值
    missing_count = (df == missing_value).sum().sum()
    if missing_count > 0:
        print(f"检测到 {missing_count} 个值为 {missing_value} 的缺失值标记，将替换为 {replace_with}")
        df.replace(missing_value, replace_with, inplace=True)
    
    # 获取目标变量和特征
    if TARGET_VARIABLE not in df.columns:
        raise ValueError(f"目标变量 '{TARGET_VARIABLE}' 不在数据列中。可用列: {df.columns.tolist()}")
    
    y = df[TARGET_VARIABLE]
    X = df.drop(columns=[TARGET_VARIABLE])
    
    # 将所有列转换为数值类型
    X = X.apply(pd.to_numeric, errors='coerce')
    
    # 检查NaN数量
    nan_count = X.isna().sum().sum()
    if nan_count > 0:
        print(f"检测到 {nan_count} 个NaN值，将替换为 {replace_with}")
        X.fillna(replace_with, inplace=True)
    
    # 检查目标变量中的NaN
    target_nan_count = y.isna().sum()
    if target_nan_count > 0:
        print(f"警告: 目标变量 '{TARGET_VARIABLE}' 中有 {target_nan_count} 个NaN值")
        print("移除目标变量为NaN的行...")
        valid_indices = ~y.isna()
        X = X[valid_indices]
        y = y[valid_indices]
        print(f"移除后的数据大小: X={X.shape}, y={y.shape}")
    
    # 查找全为零或常数的列
    constant_cols = [col for col in X.columns if X[col].nunique() <= 1]
    if constant_cols:
        print(f"移除 {len(constant_cols)} 个常数列: {constant_cols}")
        X = X.drop(columns=constant_cols)
    
    # 显示最终的特征集
    print(f"最终特征数量: {X.shape[1]}")
    print(f"目标变量统计信息: 均值={y.mean():.2f}, 最小值={y.min():.2f}, 最大值={y.max():.2f}")
    
    return X, y

# --- 评估模型函数 ---
def evaluate_model(model, X_test, y_test):
    """评估模型性能并生成可视化图表"""
    print("\n[6/7] 在测试集上进行预测并评估模型性能...")
    
    # 预测
    start_time = time.time()
    y_pred = model.predict(X_test)
    prediction_time = time.time() - start_time
    
    # 计算指标
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # 打印评估结果
    print("\n--- 模型评估结果 (测试集) ---")
    print(f"均方根误差 (RMSE): {rmse:.4f}")
    print(f"平均绝对误差 (MAE): {mae:.4f}")
    print(f"R² 分数 (决定系数): {r2:.4f}")
    print(f"预测 {len(y_test)} 个样本耗时: {prediction_time:.4f} 秒 "
          f"({len(y_test)/prediction_time:.1f} 样本/秒)")
    
    # 计算误差分布
    errors = y_test - y_pred
    print(f"误差统计 - 最小: {errors.min():.2f}, 最大: {errors.max():.2f}, "
          f"平均: {errors.mean():.2f}, 标准差: {errors.std():.2f}")
    
    # 创建散点图比较真实值与预测值
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('实际 XCO2')
    plt.ylabel('预测 XCO2')
    plt.title(f'实际 vs 预测 XCO2 (RMSE={rmse:.4f}, R²={r2:.4f})')
    
    # 保存图像
    scatter_path = os.path.splitext(MODEL_SAVE_PATH)[0] + "_predictions.png"
    plt.savefig(scatter_path, dpi=300)
    plt.close()
    
    # 创建误差直方图
    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=50)
    plt.xlabel('预测误差')
    plt.ylabel('频次')
    plt.title('预测误差分布')
    
    # 保存图像
    hist_path = os.path.splitext(MODEL_SAVE_PATH)[0] + "_errors.png"
    plt.savefig(hist_path, dpi=300)
    plt.close()
    
    return rmse, mae, r2, y_pred

# --- 主脚本 ---
if __name__ == "__main__":
    start_time = time.time()
    print("--- 开始 Deep Forest 回归模型训练 ---")
    
    # 1. 加载数据
    try:
        print(f"\n[1/7] 正在加载数据从: {FILE_PATH}")
        data = pd.read_csv(FILE_PATH)
        print(f"数据加载成功。 数据形状: {data.shape}")
        print(f"数据列: {data.columns.tolist()}")
    except FileNotFoundError:
        print(f"错误: 文件未找到 {FILE_PATH}")
        exit()
    except Exception as e:
        print(f"加载数据时出错: {e}")
        exit()
    
    # 2. 数据预处理
    X, y = preprocess_data(data, missing_value=MISSING_VALUE, replace_with=REPLACE_MISSING_WITH)
    
    # 3. 数据分割
    print(f"\n[3/7] 正在将数据拆分为训练集和测试集 (测试集比例: {TEST_SIZE})...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f"训练集大小: X={X_train.shape}, y={y_train.shape}")
    print(f"测试集大小: X={X_test.shape}, y={y_test.shape}")
    
    # 4. 初始化 Deep Forest 回归模型
    print("\n[4/7] 初始化 Deep Forest 回归模型...")
    model = CascadeForestRegressor(
        n_estimators=100,      # 森林中树的数量
        max_layers=10,         # 级联森林的最大层数
        n_jobs=-1,             # 使用所有CPU核心
        random_state=RANDOM_STATE
    )
    print("模型参数设置完成。")
    
    # 5. 训练模型
    print("\n[5/7] 开始训练模型...")
    train_start = time.time()
    model.fit(X_train.values, y_train.values)
    train_time = time.time() - train_start
    print(f"模型训练完成。训练耗时: {train_time:.2f} 秒")
    
    # 6. 评估模型
    rmse, mae, r2, y_pred = evaluate_model(model, X_test.values, y_test.values)
    
    # 7. 显示特征重要性
    print("\n[5/7] 分析特征重要性...")
    try:
        # 获取特征重要性（如果支持）
        if hasattr(model, 'feature_importances_'):
            feature_importances = pd.DataFrame({
                'feature': X.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print("\n特征重要性 (Top 20):")
            print(feature_importances.head(20))
            
            # 保存特征重要性到文件
            importance_path = os.path.splitext(MODEL_SAVE_PATH)[0] + "_feature_importances.csv"
            feature_importances.to_csv(importance_path, index=False)
            print(f"特征重要性已保存到: {importance_path}")
    except Exception as e:
        print(f"无法获取或显示特征重要性: {e}")
    
    # 8. 保存模型
    print(f"\n[7/7] 正在保存训练好的模型到: {MODEL_SAVE_PATH}")
    try:
        # 确保保存目录存在
        os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
        
        # 保存模型
        with open(MODEL_SAVE_PATH, "wb") as f:
            pickle.dump(model, f)
        print("模型已成功保存。")
        
        # 保存特征名称供预测时使用
        feature_names_path = os.path.splitext(MODEL_SAVE_PATH)[0] + "_feature_names.txt"
        with open(feature_names_path, 'w') as f:
            f.write('\n'.join(X.columns.tolist()))
        print(f"特征名称已保存到: {feature_names_path}")
    except Exception as e:
        print(f"保存模型时出错: {e}")
    
    # 计算总时间
    total_time = time.time() - start_time
    print(f"\n--- 脚本执行完毕 (用时: {total_time:.2f} 秒) ---")
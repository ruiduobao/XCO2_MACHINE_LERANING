import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np
import subprocess  # 用于调用 nvidia-smi 检查 GPU
import os
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
import joblib
import matplotlib as mpl
from matplotlib.font_manager import FontProperties
import matplotlib
matplotlib.use('Agg')  # 非交互式后端，适用于无显示器环境

# --- 配置 ---
FILE_PATH = r"E:\地理所\论文\中国XCO2论文_2025.04\数据\训练表格数据\卫星数据\处理后的网格XCO2加权统计_优化版.csv"
TARGET_VARIABLE = 'xco2'  # 目标变量列名
MODEL_SAVE_PATH = r"E:\地理所\论文\中国XCO2论文_2025.04\处理结果\模型数据\lightGBM\lightgbm_xco2_regression_model.pkl"  # 模型保存路径
TEST_SIZE = 0.2  # 测试集比例
RANDOM_STATE = 42  # 随机种子
MISSING_VALUE = -9999  # 缺失值标记
REPLACE_MISSING_WITH = 0  # 将缺失值替换为0

# 模型参数
PARAMS = {
    'objective': 'regression',
    'n_estimators': 200,
    'learning_rate': 0.1,
    'max_depth': 7,
    'num_leaves': 31,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,  # L1正则化
    'reg_lambda': 0.1,  # L2正则化
    'random_state': RANDOM_STATE,
    'n_jobs': -1,  # 使用所有CPU核心
    'verbose': -1  # 减少输出
}

# 早停参数
EARLY_STOPPING_ROUNDS = 10
EVAL_METRIC = 'rmse'

# 使用交叉验证
USE_CV = True
N_FOLDS = 5

# --- 检查 GPU 可用性 ---
def check_gpu_availability():
    """检查系统中是否有可用的 NVIDIA GPU"""
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, check=True)
        print("GPU 可用性检查：检测到 NVIDIA GPU。")
        print("注意: 使用GPU需要LightGBM是GPU编译版本。")
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("GPU 可用性检查：未检测到可用的 NVIDIA GPU。将使用 CPU。")
        return False
    except Exception as e:
        print(f"GPU 可用性检查：发生未知错误: {e}。将使用 CPU。")
        return False

# --- 数据预处理函数 ---
def preprocess_data(data, missing_value=-9999, replace_with=0):
    """
    预处理数据，处理缺失值和异常值
    
    Args:
        data (DataFrame): 输入数据
        missing_value (float): 需要替换的缺失值标记 (如-9999)
        replace_with (float): 替换缺失值的值 (如0)
        
    Returns:
        DataFrame: 处理后的数据
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

# --- 配置matplotlib支持中文 ---
def setup_chinese_font():
    """配置matplotlib支持中文显示"""
    try:
        # 尝试使用系统中的中文字体
        font_candidates = [
            'SimHei',           # 中文黑体
            'Microsoft YaHei',  # 微软雅黑
            'SimSun',           # 中文宋体
            'FangSong',         # 仿宋
            'KaiTi',            # 楷体
            'WenQuanYi Micro Hei' # Linux常见中文字体
        ]
        
        # 尝试找到可用的中文字体
        font_found = False
        for font_name in font_candidates:
            try:
                font_props = FontProperties(fname=mpl.font_manager.findfont(font_name))
                plt.rcParams['font.family'] = font_props.get_name()
                print(f"成功设置中文字体: {font_name}")
                font_found = True
                break
            except:
                continue
        
        if not font_found:
            # 如果找不到中文字体，尝试使用通用设置
            plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
            plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
            print("使用默认中文字体设置")
            
    except Exception as e:
        print(f"设置中文字体时出错: {e}")
        print("将使用matplotlib默认字体，中文可能无法正确显示")

# --- 特征重要性可视化 ---
def plot_feature_importance(model, feature_names, top_n=20, save_path=None):
    """绘制特征重要性图"""
    # 首先设置中文字体
    setup_chinese_font()
    
    try:
        # 获取特征重要性 - 对Booster对象和LGBMRegressor都适用
        if hasattr(model, 'feature_importances_'):
            # sklearn接口
            importances = model.feature_importances_
        else:
            # 原生Booster对象
            importances = model.feature_importance(importance_type='split')
        
        # 确保特征名称列表长度与特征重要性数组匹配
        if len(feature_names) != len(importances):
            print(f"警告: 特征名称列表长度({len(feature_names)})与特征重要性数组长度({len(importances)})不匹配")
            if len(feature_names) > len(importances):
                feature_names = feature_names[:len(importances)]
            else:
                # 补充缺失的特征名称
                feature_names = list(feature_names) + [f'Feature_{i}' for i in range(len(feature_names), len(importances))]
        
        indices = np.argsort(importances)[::-1]
        
        # 选取前N个特征
        n_features = min(top_n, len(feature_names))
        top_indices = indices[:n_features]
        top_features = [feature_names[i] for i in top_indices]
        top_importances = importances[top_indices]
        
        # 绘制水平条形图
        plt.figure(figsize=(10, max(6, n_features * 0.3)))
        plt.barh(range(n_features), top_importances, align='center')
        plt.yticks(range(n_features), top_features)
        plt.xlabel('特征重要性')
        plt.ylabel('特征')
        plt.title(f'LightGBM 特征重要性 (Top {n_features})')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300)
            print(f"特征重要性图已保存到: {save_path}")
    
    except Exception as e:
        print(f"绘制特征重要性时出错: {e}")
        # 记录更详细的错误信息
        import traceback
        traceback.print_exc()

# --- 交叉验证训练 ---
def train_with_cv(X, y, params, n_folds=5):
    """使用交叉验证训练模型"""
    print(f"\n正在进行 {n_folds} 折交叉验证...")
    
    # 初始化交叉验证
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
    
    # 存储每折的结果
    cv_scores = {
        'rmse': [],
        'mae': [],
        'r2': []
    }
    
    # 存储所有模型
    models = []
    
    # 进行交叉验证
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"\n--- Fold {fold+1}/{n_folds} ---")
        
        # 分割数据
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # 创建LightGBM数据集
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        # 训练模型
        local_params = params.copy()
        if 'n_estimators' in local_params:
            num_boost_round = local_params.pop('n_estimators')
        else:
            num_boost_round = 100
        
        model = lgb.train(
            local_params,
            train_data,
            num_boost_round=num_boost_round,
            valid_sets=[val_data],
            callbacks=[
                lgb.early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS),
                lgb.log_evaluation(period=20)
            ]
        )
        
        # 预测验证集
        y_pred = model.predict(X_val)
        
        # 计算指标
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        mae = mean_absolute_error(y_val, y_pred)
        r2 = r2_score(y_val, y_pred)
        
        # 记录指标
        cv_scores['rmse'].append(rmse)
        cv_scores['mae'].append(mae)
        cv_scores['r2'].append(r2)
        
        # 打印当前折的指标
        print(f"Fold {fold+1} - RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
        
        # 存储模型
        models.append(model)
    
    # 计算平均指标
    mean_rmse = np.mean(cv_scores['rmse'])
    mean_mae = np.mean(cv_scores['mae'])
    mean_r2 = np.mean(cv_scores['r2'])
    
    # 打印平均指标
    print("\n--- 交叉验证平均结果 ---")
    print(f"平均 RMSE: {mean_rmse:.4f} ± {np.std(cv_scores['rmse']):.4f}")
    print(f"平均 MAE: {mean_mae:.4f} ± {np.std(cv_scores['mae']):.4f}")
    print(f"平均 R²: {mean_r2:.4f} ± {np.std(cv_scores['r2']):.4f}")
    
    # 选择最佳模型（根据RMSE）
    best_model_idx = np.argmin(cv_scores['rmse'])
    best_model = models[best_model_idx]
    print(f"选择了第 {best_model_idx+1} 折的模型作为最佳模型 (RMSE: {cv_scores['rmse'][best_model_idx]:.4f})")
    
    return best_model, cv_scores

# --- 预测与评估函数 ---
def evaluate_model(model, X_test, y_test):
    """评估模型性能"""
    print("\n[6/7] 在测试集上进行预测并评估模型性能...")
    
    # 首先设置中文字体
    setup_chinese_font()
    
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
    plt.show()
    
    # 创建误差直方图
    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=50)
    plt.xlabel('预测误差')
    plt.ylabel('频次')
    plt.title('预测误差分布')
    
    # 保存图像
    hist_path = os.path.splitext(MODEL_SAVE_PATH)[0] + "_errors.png"
    plt.savefig(hist_path, dpi=300)
    plt.show()
    
    return rmse, mae, r2, y_pred

# --- 主脚本 ---
if __name__ == "__main__":
    start_time = time.time()
    print("--- 开始 LightGBM 回归模型训练 ---")
    print(f"LightGBM 版本: {lgb.__version__}")
    
    # 设置中文字体
    setup_chinese_font()

    # 1. 检查 GPU 可用性
    use_gpu = check_gpu_availability()
    if use_gpu:
        PARAMS['device'] = 'gpu'
    else:
        PARAMS['device'] = 'cpu'
    
    print(f"将使用 '{PARAMS['device']}' 进行训练。")

    # 2. 加载数据
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

    # 3. 数据预处理
    X, y = preprocess_data(data, missing_value=MISSING_VALUE, replace_with=REPLACE_MISSING_WITH)

    # 4. 数据分割
    print(f"\n[3/7] 正在将数据拆分为训练集和测试集 (测试集比例: {TEST_SIZE})...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f"训练集大小: X={X_train.shape}, y={y_train.shape}")
    print(f"测试集大小: X={X_test.shape}, y={y_test.shape}")

    # 5. 训练模型
    print("\n[4/7] 训练 LightGBM 回归模型...")
    
    if USE_CV:
        # 使用交叉验证训练
        model, cv_scores = train_with_cv(X_train, y_train, PARAMS, n_folds=N_FOLDS)
    else:
        # 创建LightGBM数据集
        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
        
        # 提取n_estimators参数
        if 'n_estimators' in PARAMS:
            num_boost_round = PARAMS.pop('n_estimators')
        else:
            num_boost_round = 100
        
        # 训练模型
        print("开始训练模型...")
        model = lgb.train(
            PARAMS,
            train_data,
            num_boost_round=num_boost_round,
            valid_sets=[valid_data],
            callbacks=[
                lgb.early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS),
                lgb.log_evaluation(period=20)
            ]
        )
        
        print(f"模型训练完成。最佳迭代次数: {model.best_iteration}")

    # 6. 评估模型
    rmse, mae, r2, y_pred = evaluate_model(model, X_test, y_test)

    # 7. 显示特征重要性
    print("\n[5/7] 分析特征重要性...")
    try:
        # 绘制特征重要性图
        importance_path = os.path.splitext(MODEL_SAVE_PATH)[0] + "_feature_importance.png"
        plot_feature_importance(model, X.columns, top_n=20, save_path=importance_path)
    except Exception as e:
        print(f"无法绘制特征重要性: {e}")

    # 8. 保存模型
    print(f"\n[7/7] 正在保存训练好的模型到: {MODEL_SAVE_PATH}")
    try:
        # 确保保存目录存在
        os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
        
        # 保存模型（使用joblib可以保存更大的模型）
        joblib.dump(model, MODEL_SAVE_PATH)
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

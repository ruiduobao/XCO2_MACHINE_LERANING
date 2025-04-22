import os
import sys
import argparse
import torch
import numpy as np
import rasterio
from tqdm import tqdm
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our modules
from step6_convlstm.data_loader import list_xco2_files, load_tif_file
from step6_convlstm.model import XCO2ConvLSTM, predict_multiple_steps


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Predict XCO2 for 2018 using trained ConvLSTM model')
    
    # Model parameters
    parser.add_argument('--model_path', type=str, default=r"E:\地理所\论文\中国XCO2论文_2025.04\代码\step6_convlstm\已训练模型\best_xco2_convlstm_model.pth",
                        help='Path to trained model weights (.pth file)')
    
    # Input parameters
    parser.add_argument('--data_dir', type=str, default=r'E:\地理所\论文\中国XCO2论文_2025.04\数据\XGBOOST_XCO2',
                        help='Directory containing XCO2 TIF files')
    parser.add_argument('--sequence_length', type=int, default=6,
                        help='Length of input sequences (in months)')
    parser.add_argument('--input_size', type=int, nargs=2, default=None,
                        help='Optional size to resize inputs, e.g., --input_size 128 128')
    
    # Prediction parameters
    parser.add_argument('--start_date', type=str, default="2017-12",
                        help='Start date for prediction sequence (YYYY-MM format)')
    parser.add_argument('--num_steps', type=int, default=12,
                        help='Number of steps (months) to predict')
    
    # Output parameters
    parser.add_argument('--output_dir', type=str, default=r'E:\地理所\论文\中国XCO2论文_2025.04\代码\step6_convlstm\predictions_2018',
                        help='Directory to save prediction TIFs')
    parser.add_argument('--visualize', action='store_true',
                        help='Generate visualizations of predictions')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU index to use (-1 for CPU)')
    
    return parser.parse_args()


def create_prediction_sequence(xco2_files, start_date, sequence_length, input_size=None):
    """
    创建预测序列，并填充额外通道以匹配模型期望的输入通道数。
    
    Args:
        xco2_files (list): XCO2 TIF 文件列表
        start_date (str): 'YYYY-MM' 格式的序列结束日期
        sequence_length (int): 输入序列长度
        input_size (tuple): 可选的输入大小调整
    
    Returns:
        tuple: (sequence_tensor, tif_profile) 用于预测和保存输出
    """
    # 解析起始日期
    try:
        year, month = map(int, start_date.split('-'))
        target_date = datetime(year, month, 1)
    except (ValueError, AttributeError):
        raise ValueError(f"无效的起始日期格式: {start_date}。使用 YYYY-MM 格式。")
    
    # 寻找要包含在序列中的文件
    sequence_files = []
    sorted_files = sorted(xco2_files, key=lambda x: os.path.basename(x))
    
    # 收集目标日期之前的文件
    for file_path in sorted_files:
        filename = os.path.basename(file_path)
        try:
            file_year, file_month = map(int, filename.split('_')[:2])
            file_date = datetime(file_year, file_month, 1)
            
            # 检查文件日期是否在目标日期之前或等于目标日期
            if file_date <= target_date:
                sequence_files.append(file_path)
            else:
                # 跳过目标日期之后的文件
                break
        except (ValueError, IndexError):
            print(f"警告: 跳过文件名格式无效的文件: {filename}")
    
    # 取最后 N 个文件形成序列
    if len(sequence_files) < sequence_length:
        raise ValueError(f"在 {start_date} 之前没有足够的文件来创建长度为 {sequence_length} 的序列")
    
    sequence_files = sequence_files[-sequence_length:]
    print(f"使用以下文件作为输入序列:")
    for file in sequence_files:
        print(f"  - {os.path.basename(file)}")
    
    # 加载序列数据
    sequence_data = []
    tif_profile = None
    
    for file_path in sequence_files:
        # 仅加载 XCO2 数据
        data, profile = load_tif_file(file_path, normalize=True)
        if tif_profile is None:
            tif_profile = profile
        sequence_data.append(data)
    
    # 堆叠成数组
    sequence_array = np.stack(sequence_data)
    sequence_array = sequence_array[:, np.newaxis, :, :]  # [seq_len, channels=1, height, width]
    
    # 填充额外通道以匹配模型期望的输入通道数
    model_channels = 9  # 正确的输入通道数！不是 41
    current_channels = sequence_array.shape[1]  # 通常是 1 (只有 XCO2)
    padding_channels = model_channels - current_channels
    
    if padding_channels > 0:
        # 创建全零填充数组
        padding = np.zeros((sequence_array.shape[0], padding_channels, *sequence_array.shape[2:]), dtype=sequence_array.dtype)
        # 连接原始 XCO2 数据和填充数据
        sequence_array = np.concatenate([sequence_array, padding], axis=1)
        print(f"已填充输入数据从 {current_channels} 到 {model_channels} 个通道")
    
    # 如果需要调整大小
    if input_size is not None:
        from skimage.transform import resize
        resized_array = np.zeros((sequence_array.shape[0], sequence_array.shape[1], *input_size), dtype=sequence_array.dtype)
        for i in range(sequence_array.shape[0]):
            for j in range(sequence_array.shape[1]):
                resized_array[i, j] = resize(sequence_array[i, j], input_size, 
                                             preserve_range=True, anti_aliasing=True)
        sequence_array = resized_array
        print(f"已将输入尺寸调整为 {input_size}")
    
    # 转换为 torch 张量并添加批次维度
    sequence_tensor = torch.tensor(sequence_array, dtype=torch.float32).unsqueeze(0)
    
    return sequence_tensor, tif_profile


def denormalize_prediction(prediction, reference_file):
    """
    将预测值反归一化到原始范围。
    
    Args:
        prediction (numpy.ndarray): 归一化的预测数组
        reference_file (str): 获取原始值范围的参考文件路径
        
    Returns:
        numpy.ndarray: 反归一化的预测数组
    """
    with rasterio.open(reference_file) as src:
        # 读取数据并查找最小/最大值
        reference_data = src.read(1)
        nodata_value = src.nodata
        
        # 掩盖 NoData 值
        if nodata_value is not None:
            mask = reference_data != nodata_value
            if np.any(mask):
                min_val = np.min(reference_data[mask])
                max_val = np.max(reference_data[mask])
            else:
                min_val, max_val = 0, 1  # 如果全部是 NoData，使用默认值
        else:
            min_val = np.min(reference_data)
            max_val = np.max(reference_data)
    
    # 应用反归一化: scaled_x = x * (max - min) + min
    denormalized = prediction * (max_val - min_val) + min_val
    return denormalized


def save_prediction_as_tif(prediction, profile, output_path):
    """
    将预测数组保存为 GeoTIFF 文件。
    
    Args:
        prediction (numpy.ndarray): 2D 预测数组
        profile (dict): 包含地理参考信息的配置
        output_path (str): 保存输出 TIF 文件的路径
    """
    # 更新输出的配置
    profile.update({
        'count': 1,
        'dtype': 'float32',
        'compress': 'lzw',
        'tiled': True,
        'blockxsize': 256,
        'blockysize': 256
    })
    
    # 写入文件
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(prediction.astype(np.float32), 1)


def visualize_predictions(predictions, output_dir, start_date):
    """
    生成预测结果的可视化。
    
    Args:
        predictions (list): 预测数组列表
        output_dir (str): 保存可视化的目录
        start_date (str): 'YYYY-MM' 格式的起始日期
    """
    # 创建可视化目录
    viz_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(viz_dir, exist_ok=True)
    
    # 解析起始日期
    year, month = map(int, start_date.split('-'))
    prediction_date = datetime(year, month, 1)
    
    # 创建可视化
    for i, prediction in enumerate(predictions):
        # 计算此预测的日期
        if i > 0:  # 跳过第一个预测（即起始日期）
            prediction_date = prediction_date + relativedelta(months=1)
        
        pred_date_str = prediction_date.strftime('%Y-%m')
        
        # 创建图形
        plt.figure(figsize=(10, 8))
        plt.imshow(prediction, cmap='viridis')
        plt.colorbar(label='XCO2')
        plt.title(f'{pred_date_str} 的 XCO2 预测')
        plt.tight_layout()
        
        # 保存图形
        filename = f'prediction_{pred_date_str}.png'
        plt.savefig(os.path.join(viz_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()


def main():
    """主预测脚本。"""
    # 解析参数
    args = parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 设置设备
    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
        print(f"使用 GPU: {torch.cuda.get_device_name(args.gpu)}")
    else:
        device = torch.device('cpu')
        print("使用 CPU")
    
    # 固定模型配置，与训练时一致
    hidden_dims = [32, 64]
    num_layers = len(hidden_dims)
    input_channels = 9  # 从分析脚本确认的模型输入通道数
    kernel_size = 3
    dropout = 0.2
    print(f"使用固定模型配置: 输入通道数={input_channels}, 隐藏层维度={hidden_dims}")
    
    # 初始化模型
    model = XCO2ConvLSTM(
        input_channels=input_channels,
        hidden_dims=hidden_dims,
        kernel_size=kernel_size,
        num_layers=num_layers,
        dropout=dropout
    )
    
    # 加载模型权重
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model = model.to(device)
    model.eval()
    print(f"已从 {args.model_path} 加载模型")
    
    # 列出并排序 XCO2 文件
    xco2_files = list_xco2_files(args.data_dir)
    print(f"找到 {len(xco2_files)} 个 XCO2 文件")
    
    # 确定起始日期（如果未提供）
    if args.start_date is None:
        # 使用最后可用文件的日期
        last_file = xco2_files[-1]
        filename = os.path.basename(last_file)
        try:
            year, month = map(int, filename.split('_')[:2])
            args.start_date = f"{year}-{month:02d}"
        except (ValueError, IndexError):
            print("错误: 无法从最后一个文件确定起始日期")
            return
    
    print(f"预测起始日期: {args.start_date}")
    
    # 创建输入序列（只使用 XCO2 数据，填充其余通道）
    try:
        sequence_tensor, tif_profile = create_prediction_sequence(
            xco2_files, args.start_date, args.sequence_length, args.input_size
        )
        print(f"创建了形状为 {sequence_tensor.shape} 的输入序列")
    except Exception as e:
        print(f"创建输入序列时出错: {e}")
        return
    
    # 生成预测
    predictions = predict_multiple_steps(model, sequence_tensor, args.num_steps, device)
    print(f"生成了 {len(predictions)} 个预测")
    
    # 解析起始日期用于命名输出文件
    year, month = map(int, args.start_date.split('-'))
    prediction_date = datetime(year, month, 1)
    
    # 用于反归一化的参考文件（使用序列中的最后一个文件）
    reference_file = xco2_files[-1]
    
    # 保存预测为 TIF 文件
    print("正在保存预测的 TIF 文件...")
    for i, prediction in enumerate(tqdm(predictions)):
        # 每个预测前进一个月
        prediction_date = prediction_date + relativedelta(months=1)
        pred_date_str = prediction_date.strftime('%Y_%m')
        
        # 反归一化预测
        denormalized = denormalize_prediction(prediction, reference_file)
        
        # 保存为 TIF
        output_path = os.path.join(args.output_dir, f"{pred_date_str}_CONVLSTM_XCO2.tif")
        save_prediction_as_tif(denormalized, tif_profile, output_path)
    
    print(f"已将 {len(predictions)} 个预测 TIF 保存到 {args.output_dir}")
    
    # 如果请求，生成可视化
    if args.visualize:
        print("正在生成可视化...")
        visualize_predictions(predictions, args.output_dir, args.start_date)
        print(f"可视化已保存到 {os.path.join(args.output_dir, 'visualizations')}")
    
    print("预测完成!")


if __name__ == "__main__":
    main()
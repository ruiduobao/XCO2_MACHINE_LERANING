import os
import sys
import argparse
import torch
import numpy as np
import rasterio
from tqdm import tqdm
from datetime import datetime
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt

# 添加父目录到路径以导入模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入模块
from step6_convlstm.data_loader import list_xco2_files, load_tif_file
from step6_convlstm.model import XCO2ConvLSTM

def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description='简化的XCO2预测脚本')
    
    # 模型参数
    parser.add_argument('--model_path', type=str, default=r"E:\地理所\论文\中国XCO2论文_2025.04\代码\step6_convlstm\已训练模型\best_xco2_convlstm_model.pth",
                        help='训练好的模型权重路径')
    
    # 输入参数
    parser.add_argument('--data_dir', type=str, default=r'E:\地理所\论文\中国XCO2论文_2025.04\数据\XGBOOST_XCO2',
                        help='包含XCO2 TIF文件的目录')
    parser.add_argument('--sequence_length', type=int, default=3,
                        help='输入序列长度')
    parser.add_argument('--input_size', type=int, nargs=2, default=None,
                        help='可选的输入大小调整，例如 --input_size 128 128')
    
    # 预测参数
    parser.add_argument('--start_date', type=str, default="2018-03",
                        help='预测序列起始日期（YYYY-MM格式）')
    parser.add_argument('--num_steps', type=int, default=9,
                        help='要预测的步数（月份）')
    
    # 输出参数
    parser.add_argument('--output_dir', type=str, default=r'./2018预测结果',
                        help='保存预测TIF的目录')
    parser.add_argument('--visualize', action='store_true',
                        help='生成预测的可视化')
    parser.add_argument('--gpu', type=int, default=0,
                        help='使用的GPU索引（-1表示CPU）')
    
    # 掩膜参数（新增）
    parser.add_argument('--mask_file', type=str, default=r'E:\地理所\论文\中国XCO2论文_2025.04\数据\范围数据\标准栅格_WGS84.tif',
                        help='用于定义有效区域的掩膜文件')
    parser.add_argument('--nodata_value', type=float, default=-9999.0,
                        help='输出文件的NoData值')
    
    return parser.parse_args()

def load_mask_file(mask_file_path):
    """
    加载掩膜文件，定义有效区域
    
    参数:
        mask_file_path (str): 掩膜文件路径
        
    返回:
        tuple: (mask_array, profile)
    """
    with rasterio.open(mask_file_path) as src:
        mask_data = src.read(1)
        profile = src.profile.copy()
        
        # 创建布尔掩膜 (值为1的区域被视为有效)
        valid_mask = mask_data == 1
        
    return valid_mask, profile

def main():
    """主函数。"""
    args = parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 设置设备
    device = torch.device(f'cuda:{args.gpu}' if args.gpu >= 0 and torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 初始化和加载模型
    model = XCO2ConvLSTM(
        input_channels=9,  # 使用正确的输入通道数
        hidden_dims=[32, 64],
        kernel_size=3,
        num_layers=2,
        dropout=0.2
    )
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval()
    print(f"已从 {args.model_path} 加载模型")
    
    # 列出和排序文件
    files = list_xco2_files(args.data_dir)
    print(f"找到 {len(files)} 个XCO2文件")
    
    # 确定起始和结束日期
    year, month = map(int, args.start_date.split('-'))
    start_date = datetime(year, month, 1)
    
    # 找出对应的输入文件
    input_files = []
    for file in files:
        filename = os.path.basename(file)
        file_year, file_month = map(int, filename.split('_')[:2])
        file_date = datetime(file_year, file_month, 1)
        
        if file_date <= start_date:
            input_files.append(file)
    
    # 只使用最后几个文件作为输入
    input_files = input_files[-args.sequence_length:]
    if len(input_files) < args.sequence_length:
        print(f"警告: 只找到 {len(input_files)} 个文件，少于请求的序列长度 {args.sequence_length}")
    
    print("使用以下文件作为输入:")
    for file in input_files:
        print(f"  - {os.path.basename(file)}")
    
    # 加载掩膜文件
    print(f"正在加载掩膜文件: {args.mask_file}")
    if not os.path.exists(args.mask_file):
        print(f"错误: 掩膜文件不存在: {args.mask_file}")
        return
    
    valid_region_mask, mask_profile = load_mask_file(args.mask_file)
    print(f"掩膜文件加载成功，有效区域像素数: {np.sum(valid_region_mask)}")
    
    # 加载输入数据
    input_data = []
    input_masks = []
    profile = None
    
    for file in input_files:
        data, data_mask, p = load_tif_file(file, normalize=True, preserve_mask=True)
        if profile is None:
            profile = p
        input_data.append(data)
        input_masks.append(data_mask)
    
    # 组合所有文件的掩膜以创建一个统一的有效区域掩膜
    # 进一步与区域掩膜相交，确保只预测所需的区域
    combined_input_mask = np.logical_and.reduce(input_masks)
    final_valid_mask = np.logical_and(combined_input_mask, valid_region_mask)
    print(f"组合后的有效掩膜像素数: {np.sum(final_valid_mask)}")
    
    # 从加载的数据创建序列张量
    input_sequence = np.stack(input_data)[:, np.newaxis, :, :]  # [seq_len, channels=1, height, width]
    
    # 填充到9个通道
    padding = np.zeros((input_sequence.shape[0], 8, *input_sequence.shape[2:]), dtype=input_sequence.dtype)
    input_sequence = np.concatenate([input_sequence, padding], axis=1)
    print(f"填充输入数据到 9 个通道，形状: {input_sequence.shape}")
    
    # 如果需要调整大小
    if args.input_size:
        from skimage.transform import resize
        
        # 调整输入序列大小
        resized = np.zeros((input_sequence.shape[0], input_sequence.shape[1], *args.input_size), dtype=input_sequence.dtype)
        for i in range(input_sequence.shape[0]):
            for j in range(input_sequence.shape[1]):
                resized[i, j] = resize(input_sequence[i, j], args.input_size, preserve_range=True, anti_aliasing=True)
        input_sequence = resized
        
        # 调整掩膜大小
        resized_mask = resize(final_valid_mask.astype(np.float32), args.input_size, 
                              preserve_range=True, anti_aliasing=False) > 0.5
        final_valid_mask = resized_mask
        
        print(f"已调整输入和掩膜大小为 {args.input_size}")
    
    # 转换为张量并添加批次维度
    input_tensor = torch.tensor(input_sequence, dtype=torch.float32).unsqueeze(0).to(device)
    
    # 预测
    predictions = []
    masks = []
    current_input = input_tensor
    
    print(f"开始预测 {args.num_steps} 个时间步...")
    with torch.no_grad():
        for step in range(args.num_steps):
            # 进行一次前向传播
            output = model(current_input)
            
            # 将输出保存为numpy数组
            prediction = output.squeeze().cpu().numpy()
            
            # 应用掩膜 - 只保留有效区域的预测值
            masked_prediction = np.where(final_valid_mask, prediction, 0)
            
            predictions.append(masked_prediction)
            masks.append(final_valid_mask.copy())
            
            # 创建新的输入序列
            # 移除最早的时间步
            new_seq = current_input[:, 1:].clone()
            
            # 创建新的时间步（保持9个通道的结构）
            # 首先创建全零时间步
            new_step = torch.zeros((1, 1, 9, *current_input.shape[3:]), device=device)
            
            # 将预测放入第一个通道，但保留掩膜效果
            new_step[0, 0, 0] = torch.tensor(masked_prediction, device=device)
            
            # 连接
            current_input = torch.cat([new_seq, new_step], dim=1)
            
            month_date = start_date + relativedelta(months=step+1)
            print(f"已预测 {month_date.strftime('%Y-%m')}")
    
    # 保存预测
    print("正在保存预测...")
    for i, (pred, mask) in enumerate(zip(predictions, masks)):
        pred_date = start_date + relativedelta(months=i+1)
        date_str = pred_date.strftime('%Y_%m')
        
        # 反归一化
        with rasterio.open(files[-1]) as src:
            data = src.read(1)
            nodata = src.nodata
            if nodata is not None:
                data_mask = data != nodata
                min_val = np.min(data[data_mask])
                max_val = np.max(data[data_mask])
            else:
                min_val = np.min(data)
                max_val = np.max(data)
        
        # 对有效区域进行反归一化，无效区域设为NoData值
        denormalized = np.full_like(pred, args.nodata_value, dtype=np.float32)
        denormalized[mask] = pred[mask] * (max_val - min_val) + min_val
        
        # 保存为TIF
        out_path = os.path.join(args.output_dir, f"{date_str}_CONVLSTM_XCO2.tif")
        profile.update({
            'count': 1,
            'dtype': 'float32',
            'compress': 'lzw',
            'tiled': True,
            'nodata': args.nodata_value
        })
        
        with rasterio.open(out_path, 'w', **profile) as dst:
            dst.write(denormalized.astype(np.float32), 1)
    
    print(f"已将 {len(predictions)} 个预测保存到 {args.output_dir}")
    
    # 如果需要可视化
    if args.visualize:
        viz_dir = os.path.join(args.output_dir, 'visualizations')
        os.makedirs(viz_dir, exist_ok=True)
        
        # 创建自定义colormap，将NoData值显示为透明
        cmap = plt.cm.get_cmap('viridis').copy()
        cmap.set_bad('white', alpha=0)
        
        for i, (pred, mask) in enumerate(zip(predictions, masks)):
            pred_date = start_date + relativedelta(months=i+1)
            date_str = pred_date.strftime('%Y-%m')
            
            # 创建掩膜版本的预测，以便正确显示NoData区域
            masked_viz = np.ma.array(pred, mask=~mask)
            
            plt.figure(figsize=(10, 8))
            plt.imshow(masked_viz, cmap=cmap)
            plt.colorbar(label='XCO2')
            plt.title(f'{date_str} 的 XCO2 预测')
            plt.tight_layout()
            
            plt.savefig(os.path.join(viz_dir, f'prediction_{date_str}.png'), dpi=300)
            plt.close()
        
        print(f"可视化已保存到 {viz_dir}")
    
    print("预测完成!")

if __name__ == "__main__":
    main()
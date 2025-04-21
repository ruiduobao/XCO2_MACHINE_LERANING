import os
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
import torch
import torch.nn as nn
import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import OrderedDict
import time
import concurrent.futures
from functools import partial

# --- 配置 ---
# 模型路径
MODEL_PATH = r"E:\地理所\论文\中国XCO2论文_2025.04\处理结果\模型数据\ST_RESNET\st_resnet_xco2_pytorch_model.pth"
# TIF文件目录
ROOT_DIR = r"E:\地理所\论文\中国XCO2论文_2025.04\数据"
# 预测年月
PREDICT_YEAR = 2018
PREDICT_MONTH = 3
# 标准栅格文件
STANDARD_GRID_PATH = r"E:\地理所\论文\中国XCO2论文_2025.04\数据\范围数据\标准栅格_WGS84.tif"
# 输出预测结果TIF
OUTPUT_TIF_PATH = r"E:\地理所\论文\中国XCO2论文_2025.04\数据\训练表格数据\标准栅格数据\标准栅格XY_添加tif_预测XCO2\预测xco2_st_resnet_pytorch_2018_03_epoch30.tif"
# 批处理大小
BATCH_SIZE = 1024
# 并行线程数
NUM_WORKERS = 4

# 数据类型映射
type_to_folder = {
    'Lantitude': '纬度栅格',
    'Longtitude': '经度栅格',
    'UnixTime': '每月时间戳的栅格数据',
    'aspect': '坡向数据',
    'slope': '坡度数据',
    'DEM': 'DEM',
    'VIIRS': '夜光遥感',
    'ERA5Land': 'ERA5',
    'AOD': '气溶胶厚度',
    'CT2019B': 'carbon_tracer',
    'landscan': 'landscan',
    'odiac1km': 'odiac',
    'humanfootprint': '人类足迹数据',
    'OCO2GEOS': 'OCO2_GEOS_XCO2同化数据',
    'CAMStcco2': 'CAMS',
    'CLCD': 'CLCD',
    'MODISLANDCOVER': 'modis_landcover',
    'MOD13A2': 'NDVI',
}

# 年度数据和月度数据分类
annual_types = ['Lantitude', 'Longtitude', 'aspect', 'slope', 'DEM', 'landscan', 'humanfootprint', 'CLCD', 'MODISLANDCOVER']
monthly_types = ['UnixTime', 'VIIRS', 'ERA5Land', 'AOD', 'CT2019B', 'odiac1km', 'OCO2GEOS', 'CAMStcco2', 'MOD13A2']

# --- 检查GPU可用性 ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# TIF数据缓存
tif_data_cache = {}

# --- 模型定义 ---
class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate=32, num_layers=4):
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer = nn.Sequential(
                nn.BatchNorm2d(in_channels + i * growth_rate),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels + i * growth_rate, growth_rate, kernel_size=3, padding=1)
            )
            self.layers.append(layer)
    
    def forward(self, x):
        features = [x]
        for layer in self.layers:
            new_feature = layer(torch.cat(features, 1))
            features.append(new_feature)
        return torch.cat(features, 1)

class IdentityBlock(nn.Module):
    def __init__(self, in_channels, filters):
        super(IdentityBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, filters, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(filters)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(filters, filters, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(filters)
    
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # 如果输入和输出通道数不同，需要使用1x1卷积调整identity
        if identity.shape[1] != out.shape[1]:
            identity = nn.Conv2d(identity.shape[1], out.shape[1], kernel_size=1)(identity)
        
        out += identity
        out = self.relu(out)
        
        return out

class STResNet(nn.Module):
    def __init__(self, in_channels, patch_size=3, growth_rate=32, block_layers=4, 
                 num_dense_blocks=2, num_identity_blocks=2):
        super(STResNet, self).__init__()
        
        # 初始卷积层
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        # 密集块
        self.dense_blocks = nn.ModuleList()
        channels = 64
        for i in range(num_dense_blocks):
            block = DenseBlock(channels, growth_rate, block_layers)
            channels += block_layers * growth_rate
            self.dense_blocks.append(block)
        
        # 恒等块
        self.identity_blocks = nn.ModuleList()
        for i in range(num_identity_blocks):
            self.identity_blocks.append(IdentityBlock(channels, channels))
        
        # 计算最终特征尺寸
        feature_size = patch_size * patch_size * channels
        
        # 全连接层
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feature_size, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )
    
    def forward(self, x):
        # 初始特征提取
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        # 通过密集块
        for block in self.dense_blocks:
            x = block(x)
        
        # 通过恒等块
        for block in self.identity_blocks:
            x = block(x)
        
        # 通过全连接层获得输出
        x = self.fc_layers(x)
        
        return x

# --- 辅助函数 ---
def read_tif_with_cache(file_path):
    """带缓存的TIF文件读取函数"""
    if file_path not in tif_data_cache:
        try:
            with rasterio.open(file_path) as src:
                # 读取整个TIF文件到内存
                data = src.read().astype(np.float32)
                
                # 将NoData值替换为NaN
                for band_idx in range(data.shape[0]):
                    if src.nodatavals and src.nodatavals[band_idx] is not None:
                        data[band_idx][data[band_idx] == src.nodatavals[band_idx]] = np.nan
                    # 显式处理-9999值
                    data[band_idx][data[band_idx] == -9999] = np.nan
                
                # 将无穷大值替换为NaN
                data[np.isinf(data)] = np.nan
                
                tif_data_cache[file_path] = {
                    'data': data,
                    'height': src.height,
                    'width': src.width
                }
        except Exception as e:
            print(f"缓存TIF文件时出错: {os.path.basename(file_path)}, 错误: {e}")
            return None
    
    return tif_data_cache[file_path]

def get_required_tifs(year, month):
    """获取指定年月需要的所有TIF文件"""
    required_tifs = {}
    
    # 处理年度数据
    for data_type in annual_types:
        folder_name = type_to_folder[data_type]
        file_path = os.path.join(ROOT_DIR, folder_name, f"{data_type}_{year}.tif")
        if os.path.exists(file_path):
            key = f"{data_type}_{year}"
            required_tifs[key] = file_path
    
    # 处理月度数据
    for data_type in monthly_types:
        folder_name = type_to_folder[data_type]
        file_path = os.path.join(ROOT_DIR, folder_name, f"{data_type}_{year}_{month:02d}.tif")
        if os.path.exists(file_path):
            key = f"{data_type}_{year}_{month:02d}"
            required_tifs[key] = file_path
    
    print(f"找到 {len(required_tifs)} 个需要的TIF文件")
    return required_tifs

def read_tif_patch(file_path, x, y, patch_size=3):
    """读取指定坐标周围的图像块（使用缓存）"""
    half_patch = patch_size // 2
    
    try:
        # 获取缓存的TIF数据
        tif_data = read_tif_with_cache(file_path)
        if tif_data is None:
            return np.full((1, patch_size, patch_size), np.nan, dtype=np.float32)
        
        data = tif_data['data']
        height = tif_data['height']
        width = tif_data['width']
        
        # 计算窗口边界，确保不超出栅格范围
        row_start = max(0, y - half_patch)
        col_start = max(0, x - half_patch)
        row_stop = min(height, y + half_patch + 1)
        col_stop = min(width, x + half_patch + 1)
        
        # 提取数据块
        patch = data[:, row_start:row_stop, col_start:col_stop]
        
        # 如果窗口尺寸小于预期（在边界），则填充到指定尺寸
        if patch.shape[1] < patch_size or patch.shape[2] < patch_size:
            padded_data = np.full((patch.shape[0], patch_size, patch_size), np.nan, dtype=np.float32)
            padded_data[:, :patch.shape[1], :patch.shape[2]] = patch
            patch = padded_data
            
        return patch
    except Exception as e:
        print(f"读取TIF文件时出错: {os.path.basename(file_path)}, 错误: {e}")
        # 如果发生错误，返回NaN填充的数组
        return np.full((1, patch_size, patch_size), np.nan, dtype=np.float32)

def get_features_for_pixel(tif_files, x, y, year, month, patch_size=3):
    """获取指定像素的所有特征"""
    feature_list = []
    
    # 收集年度数据
    for data_type in annual_types:
        key = f"{data_type}_{year}"
        if key in tif_files:
            data = read_tif_patch(tif_files[key], x, y, patch_size)
            feature_list.append(data)
    
    # 收集月度数据
    for data_type in monthly_types:
        key = f"{data_type}_{year}_{month:02d}"
        if key in tif_files:
            data = read_tif_patch(tif_files[key], x, y, patch_size)
            feature_list.append(data)
    
    # 如果没有获取到任何数据，返回空数组
    if not feature_list:
        return None
    
    # 合并所有特征
    try:
        features = np.concatenate(feature_list, axis=0)
        
        # 用均值填充NaN
        for i in range(features.shape[0]):
            band = features[i]
            mask = np.isnan(band)
            if mask.all():
                # 如果全部是NaN，填充0
                features[i][mask] = 0
            else:
                # 用非NaN值的均值填充
                features[i][mask] = np.nanmean(band)
        
        return features
    except Exception as e:
        print(f"处理特征时出错: {e}")
        return None

def adapt_features(features, target_channels):
    """调整特征维度以匹配模型需求"""
    current_channels = features.shape[0]
    
    if current_channels == target_channels:
        return features
    
    if current_channels < target_channels:
        # 如果维度不足，进行填充
        padding = np.zeros((target_channels - current_channels, features.shape[1], features.shape[2]), dtype=np.float32)
        return np.vstack([features, padding])
    else:
        # 如果维度过多，进行截断
        return features[:target_channels]

def load_model(model_path, in_channels, device):
    """加载模型"""
    print(f"正在加载模型: {model_path}")
    
    # 创建模型实例
    model = STResNet(
        in_channels=in_channels,
        patch_size=3,
        growth_rate=32,
        block_layers=4,
        num_dense_blocks=2,
        num_identity_blocks=2
    ).to(device)
    
    # 加载预训练权重
    try:
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"模型加载成功，来自epoch {checkpoint['epoch']}，验证损失: {checkpoint['val_loss']:.4f}")
        return model
    except Exception as e:
        print(f"加载模型时出错: {e}")
        return None

def preload_tif_files(tif_files):
    """预先加载所有TIF文件到内存"""
    print("预加载TIF文件到内存...")
    
    file_paths = list(tif_files.values())
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        list(tqdm(executor.map(read_tif_with_cache, file_paths), 
                  total=len(file_paths), 
                  desc="加载TIF文件"))

# --- 主函数 ---
if __name__ == "__main__":
    start_time = time.time()
    print(f"--- 开始使用PyTorch ST-ResNet模型进行XCO2预测 ---")
    print(f"预测年月: {PREDICT_YEAR}-{PREDICT_MONTH:02d}")
    
    # 1. 获取需要的TIF文件
    print("\n[1/6] 正在准备所需的TIF文件...")
    tif_files = get_required_tifs(PREDICT_YEAR, PREDICT_MONTH)
    if not tif_files:
        print("错误: 未找到需要的TIF文件，无法继续预测。")
        exit()
    
    # 2. 打开标准栅格
    print("\n[2/6] 正在打开标准栅格文件...")
    try:
        with rasterio.open(STANDARD_GRID_PATH) as src:
            grid_data = src.read(1)
            profile = src.profile.copy()
            print(f"标准栅格形状: {grid_data.shape}")
            print(f"有效像素数量: {np.sum(grid_data == 1)}")
    except Exception as e:
        print(f"打开标准栅格文件时出错: {e}")
        exit()
    
    # 3. 预加载所有TIF文件
    print("\n[3/6] 预加载所有TIF文件到内存...")
    preload_tif_files(tif_files)
    
    # 4. 为第一个像素获取特征，以确定特征数量
    print("\n[4/6] 确定特征维度...")
    valid_pixels = np.where(grid_data == 1)
    if len(valid_pixels[0]) == 0:
        print("错误: 标准栅格中没有有效像素（值为1）。")
        exit()
    
    # 获取模型需要的输入通道数
    in_channels_model = None
    try:
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        if 'model_state_dict' in checkpoint:
            model_state = checkpoint['model_state_dict']
            if 'conv1.weight' in model_state:
                in_channels_model = model_state['conv1.weight'].shape[1]
                print(f"模型需要的输入通道数: {in_channels_model}")
    except Exception as e:
        print(f"读取模型通道数时出错: {e}")
    
    # 获取样本特征维度
    first_y, first_x = valid_pixels[0][0], valid_pixels[1][0]
    sample_features = get_features_for_pixel(tif_files, first_x, first_y, PREDICT_YEAR, PREDICT_MONTH)
    
    if sample_features is None:
        print("错误: 无法为样本像素获取特征。")
        exit()
    
    in_channels_data = sample_features.shape[0]
    print(f"数据特征维度: {in_channels_data}")
    
    # 确定输入通道数
    in_channels = in_channels_model if in_channels_model is not None else in_channels_data
    
    # 5. 加载模型
    print("\n[5/6] 加载ST-ResNet模型...")
    model = load_model(MODEL_PATH, in_channels, device)
    if model is None:
        print("错误: 无法加载模型，预测终止。")
        exit()
    
    # 设置为评估模式
    model.eval()
    
    # 6. 开始预测并直接保存为TIF
    print("\n[6/6] 开始批量预测并保存为TIF...")
    # 创建输出栅格数组，使用-9999作为NoData值
    output_data = np.full_like(grid_data, -9999, dtype=np.float32)
    
    # 获取所有有效像素的坐标
    valid_indices = np.where(grid_data == 1)
    num_pixels = len(valid_indices[0])
    
    # 使用批处理进行预测
    with torch.no_grad():
        # 计算需要多少批次
        num_batches = int(np.ceil(num_pixels / BATCH_SIZE))
        
        # 统计处理像素数
        processed = 0
        errors = 0
        
        for batch_idx in tqdm(range(num_batches), desc="批次预测进度"):
            start_idx = batch_idx * BATCH_SIZE
            end_idx = min((batch_idx + 1) * BATCH_SIZE, num_pixels)
            current_batch_size = end_idx - start_idx
            
            # 为当前批次准备特征
            batch_features = []
            batch_positions = []
            
            for i in range(start_idx, end_idx):
                y, x = valid_indices[0][i], valid_indices[1][i]
                
                try:
                    # 获取此像素的特征
                    features = get_features_for_pixel(tif_files, x, y, PREDICT_YEAR, PREDICT_MONTH)
                    
                    if features is not None:
                        # 调整特征维度以匹配模型需求
                        features = adapt_features(features, in_channels)
                        batch_features.append(features)
                        batch_positions.append((y, x))
                except Exception as e:
                    errors += 1
                    if errors < 10:  # 只显示前10个错误
                        print(f"处理像素({x},{y})时出错: {e}")
            
            if not batch_features:
                continue
                
            # 堆叠批次特征
            try:
                stacked_features = np.stack(batch_features)
                tensor_features = torch.tensor(stacked_features, dtype=torch.float32).to(device)
                
                # 如果有NaN，替换为0
                tensor_features[torch.isnan(tensor_features)] = 0
                
                # 批次预测
                predictions = model(tensor_features).cpu().numpy().flatten()
                
                # 将预测结果写入输出数组
                valid_count = 0
                for idx, (y, x) in enumerate(batch_positions):
                    xco2_value = predictions[idx]
                    # 检查合理性
                    if not np.isnan(xco2_value) and 300 <= xco2_value <= 500:
                        output_data[y, x] = xco2_value
                        valid_count += 1
                
                processed += valid_count
                
                # 每20个批次报告一下进度
                if batch_idx % 20 == 0 or batch_idx == num_batches - 1:
                    elapsed = time.time() - start_time
                    pixels_per_second = processed / elapsed if elapsed > 0 else 0
                    remaining = (num_pixels - processed) / pixels_per_second if pixels_per_second > 0 else 0
                    print(f"已处理 {processed}/{num_pixels} 像素 ({processed/num_pixels*100:.1f}%), "
                          f"速度: {pixels_per_second:.1f} 像素/秒, "
                          f"预计剩余时间: {remaining/60:.1f} 分钟")
                
            except Exception as e:
                print(f"批次 {batch_idx+1}/{num_batches} 处理时出错: {e}")
    
    # 保存结果栅格
    print(f"\n预测完成，共预测 {processed}/{num_pixels} 个有效像素，出现 {errors} 个错误。")
    print(f"正在保存结果栅格到: {OUTPUT_TIF_PATH}")
    
    # 更新输出栅格配置
    profile.update(
        dtype='float32',
        count=1,
        compress='lzw',
        nodata=-9999
    )
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(OUTPUT_TIF_PATH), exist_ok=True)
    
    # 保存栅格
    with rasterio.open(OUTPUT_TIF_PATH, 'w', **profile) as dst:
        dst.write(output_data, 1)
    
    end_time = time.time()
    total_time = end_time - start_time
    print(f"预测完成！总用时: {total_time:.2f} 秒 ({total_time/60:.2f} 分钟)")
    print(f"平均处理速度: {processed/total_time:.2f} 像素/秒")
    print(f"结果已保存到: {OUTPUT_TIF_PATH}")
    
    # 在同一目录下保存预测结果的CSV文件（可选，用于兼容性）
    csv_output_path = OUTPUT_TIF_PATH.replace('.tif', '.csv')
    print(f"正在保存预测结果的CSV文件到: {csv_output_path}")
    
    # 创建结果DataFrame
    result_data = []
    for i in range(num_pixels):
        y, x = valid_indices[0][i], valid_indices[1][i]
        if output_data[y, x] != -9999:  # 只保存有效预测结果
            result_data.append({
                'X': x,
                'Y': y,
                'predicted_xco2': output_data[y, x]
            })
    
    result_df = pd.DataFrame(result_data)
    result_df.to_csv(csv_output_path, index=False)
    print(f"CSV文件保存完成，包含 {len(result_df)} 行预测结果。")
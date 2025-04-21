import os
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
from tqdm import tqdm
import glob
from collections import OrderedDict

# --- 配置 ---
# TIF文件目录
ROOT_DIR = r"E:\地理所\论文\中国XCO2论文_2025.04\数据"
# XCO2标签数据
LABEL_FILE = r"E:\地理所\论文\中国XCO2论文_2025.04\数据\训练表格数据\卫星数据\网格XCO2加权统计_按年月.csv"
# 模型保存路径
MODEL_SAVE_PATH = r"E:\地理所\论文\中国XCO2论文_2025.04\处理结果\模型数据\ST_RESNET\st_resnet_xco2_pytorch_model.pth"
# 训练参数
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001
EARLY_STOP_PATIENCE = 10
TEST_SIZE = 0.05
RANDOM_STATE = 42

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

# --- 数据集定义 ---
class XCO2Dataset(Dataset):
    def __init__(self, labels_df, root_dir, use_patches=True, patch_size=3, transform=None):
        """
        Args:
            labels_df (pandas.DataFrame): 包含XCO2标签的DataFrame
            root_dir (string): TIF文件的根目录
            use_patches (bool): 是否使用图像块而不是单个像素
            patch_size (int): 图像块的大小（边长）
            transform (callable, optional): 应用于样本的可选转换
        """
        self.labels_df = labels_df
        self.root_dir = root_dir
        self.transform = transform
        self.use_patches = use_patches
        self.patch_size = patch_size
        self.half_patch = patch_size // 2
        
        # 初始化缓存TIF文件信息
        self.tif_cache = {}
        
        # 提前标识会使用到的TIF文件
        self.required_tifs = self._get_required_tifs()
        
        # 预加载标签
        self.xco2_values = labels_df['xco2'].values
        self.coordinates = labels_df[['X', 'Y', 'year', 'month']].values
        
    def _get_required_tifs(self):
        """确定需要的TIF文件列表"""
        required_tifs = {}
        unique_years = self.labels_df['year'].unique()
        unique_year_months = self.labels_df[['year', 'month']].drop_duplicates().values
        
        # 处理年度数据
        for data_type in annual_types:
            folder_name = type_to_folder[data_type]
            for year in unique_years:
                file_path = os.path.join(self.root_dir, folder_name, f"{data_type}_{year}.tif")
                if os.path.exists(file_path):
                    key = f"{data_type}_{year}"
                    required_tifs[key] = file_path
        
        # 处理月度数据
        for data_type in monthly_types:
            folder_name = type_to_folder[data_type]
            for year, month in unique_year_months:
                file_path = os.path.join(self.root_dir, folder_name, f"{data_type}_{year}_{month:02d}.tif")
                if os.path.exists(file_path):
                    key = f"{data_type}_{year}_{month:02d}"
                    required_tifs[key] = file_path
        
        print(f"已找到 {len(required_tifs)} 个需要的TIF文件")
        return required_tifs
    
    def _get_tif_data(self, file_path, x, y):
        """读取特定坐标的TIF数据，处理缺失值"""
        if file_path not in self.tif_cache:
            try:
                with rasterio.open(file_path) as src:
                    is_time_data = 'UnixTime' in file_path
                    
                    # 获取数据类型信息
                    dtype = src.dtypes[0]
                    is_integer_type = 'int' in dtype
                    nodata_value = src.nodata
                    
                    if self.use_patches:
                        # 计算窗口边界，确保不超出栅格范围
                        row_start = max(0, y - self.half_patch)
                        col_start = max(0, x - self.half_patch)
                        row_stop = min(src.height, y + self.half_patch + 1)
                        col_stop = min(src.width, x + self.half_patch + 1)
                        window = Window(col_start, row_start, col_stop - col_start, row_stop - row_start)
                        
                        # 对于UnixTime和整数类型的TIF文件，我们需要特殊处理
                        if is_time_data or is_integer_type:
                            # 读取并立即转换为浮点类型
                            data = src.read(window=window).astype(np.float32)
                        else:
                            data = src.read(window=window)
                        
                        # 如果窗口尺寸小于预期（在边界），则填充到指定尺寸
                        if data.shape[1] < self.patch_size or data.shape[2] < self.patch_size:
                            padded_data = np.full((data.shape[0], self.patch_size, self.patch_size), np.nan, dtype=np.float32)
                            padded_data[:, :data.shape[1], :data.shape[2]] = data
                            data = padded_data
                    else:
                        # 读取单个像素值
                        if 0 <= y < src.height and 0 <= x < src.width:
                            # 对于UnixTime和整数类型的TIF文件，我们需要特殊处理
                            if is_time_data or is_integer_type:
                                # 读取并立即转换为浮点类型
                                data = src.read(window=Window(x, y, 1, 1)).astype(np.float32)
                            else:
                                data = src.read(window=Window(x, y, 1, 1))
                        else:
                            data = np.full((src.count, 1, 1), np.nan, dtype=np.float32)
                    
                    # 将NoData值替换为NaN
                    for band_idx in range(data.shape[0]):
                        if src.nodatavals and src.nodatavals[band_idx] is not None:
                            data[band_idx][data[band_idx] == src.nodatavals[band_idx]] = np.nan
                        # 显式处理-9999值
                        data[band_idx][data[band_idx] == -9999] = np.nan
                    
                    # 将无穷大值替换为NaN
                    data[np.isinf(data)] = np.nan
                    
                    # 对于时间戳数据的特殊处理
                    if is_time_data:
                        # 确保时间戳数据使用float32类型，以避免整数转换问题
                        data = data.astype(np.float32)
                
                self.tif_cache[file_path] = data
            except Exception as e:
                print(f"读取TIF文件时出错: {file_path}, 错误: {e}")
                if self.use_patches:
                    # 如果发生错误，使用NaN填充
                    self.tif_cache[file_path] = np.full((1, self.patch_size, self.patch_size), np.nan, dtype=np.float32)
                else:
                    self.tif_cache[file_path] = np.full((1, 1, 1), np.nan, dtype=np.float32)
        
        return self.tif_cache[file_path].copy()
    
    def __len__(self):
        return len(self.labels_df)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # 获取坐标和时间信息
        x, y, year, month = self.coordinates[idx]
        x, y, year, month = int(x), int(y), int(year), int(month)
        
        # 收集数据特征
        feature_list = []
        
        # 收集年度数据
        for data_type in annual_types:
            key = f"{data_type}_{year}"
            if key in self.required_tifs:
                data = self._get_tif_data(self.required_tifs[key], x, y)
                feature_list.append(data)
        
        # 收集月度数据
        for data_type in monthly_types:
            key = f"{data_type}_{year}_{month:02d}"
            if key in self.required_tifs:
                data = self._get_tif_data(self.required_tifs[key], x, y)
                feature_list.append(data)
        
        # 如果没有获取到任何数据，创建空的特征
        if not feature_list:
            if self.use_patches:
                feature_tensor = torch.full((1, self.patch_size, self.patch_size), float('nan'))
            else:
                feature_tensor = torch.full((1, 1, 1), float('nan'))
        else:
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
                
                # 转换为张量
                feature_tensor = torch.tensor(features, dtype=torch.float32)
            except Exception as e:
                print(f"处理特征时出错: {e}")
                if self.use_patches:
                    feature_tensor = torch.full((1, self.patch_size, self.patch_size), 0.0)
                else:
                    feature_tensor = torch.full((1, 1, 1), 0.0)
        
        # 获取标签
        label = torch.tensor(self.xco2_values[idx], dtype=torch.float32)
        
        # 应用变换
        if self.transform:
            feature_tensor = self.transform(feature_tensor)
        
        return feature_tensor, label

# --- 定义ST-ResNet模型 ---
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
def load_and_prepare_data(label_file, test_size=0.2, random_state=42):
    """加载并准备训练和测试数据"""
    # 加载标签数据
    labels_df = pd.read_csv(label_file)
    
    # 确保坐标列是整数类型
    labels_df['X'] = labels_df['X'].astype(int)
    labels_df['Y'] = labels_df['Y'].astype(int)
    
    # 处理缺失值
    if 'xco2' in labels_df.columns:
        # 移除xco2为NaN的行
        labels_df = labels_df.dropna(subset=['xco2'])
        # 替换-9999值
        labels_df['xco2'] = labels_df['xco2'].replace(-9999, np.nan).dropna()
    
    # 拆分训练集和测试集
    train_df, test_df = train_test_split(
        labels_df, test_size=test_size, random_state=random_state
    )
    
    print(f"训练集大小: {len(train_df)}, 测试集大小: {len(test_df)}")
    
    return train_df, test_df

def train_model(model, train_loader, val_loader, criterion, optimizer, device, 
               num_epochs=100, patience=10, model_save_path='model.pth'):
    """训练模型并保存最佳模型"""
    # 初始化最佳模型跟踪
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    # 记录训练历史
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_rmse': []
    }
    
    # 训练循环
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_steps = 0
        
        for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # 如果有NaN，替换为0
            inputs[torch.isnan(inputs)] = 0
            
            # 清除梯度
            optimizer.zero_grad()
            
            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), targets)
            
            # 反向传播与优化
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_steps += 1
        
        # 计算平均训练损失
        avg_train_loss = train_loss / train_steps
        history['train_loss'].append(avg_train_loss)
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_steps = 0
        all_targets = []
        all_predictions = []
        
        with torch.no_grad():
            for inputs, targets in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
                inputs, targets = inputs.to(device), targets.to(device)
                
                # 如果有NaN，替换为0
                inputs[torch.isnan(inputs)] = 0
                
                # 前向传播
                outputs = model(inputs)
                loss = criterion(outputs.squeeze(), targets)
                
                val_loss += loss.item()
                val_steps += 1
                
                # 收集预测结果用于计算指标
                all_targets.extend(targets.cpu().numpy())
                all_predictions.extend(outputs.squeeze().cpu().numpy())
        
        # 计算平均验证损失
        avg_val_loss = val_loss / val_steps
        history['val_loss'].append(avg_val_loss)
        
        # 计算RMSE
        val_rmse = np.sqrt(mean_squared_error(all_targets, all_predictions))
        history['val_rmse'].append(val_rmse)
        
        # 输出进度
        print(f"Epoch {epoch+1}/{num_epochs} - "
              f"Train Loss: {avg_train_loss:.4f}, "
              f"Val Loss: {avg_val_loss:.4f}, "
              f"Val RMSE: {val_rmse:.4f}")
        
        # 检查是否是最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # 保存最佳模型状态
            best_model_state = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                'val_rmse': val_rmse
            }
            print(f"找到新的最佳模型，验证损失: {best_val_loss:.4f}")
        else:
            patience_counter += 1
            print(f"验证损失未改善，耐心计数: {patience_counter}/{patience}")
        
        # 提前停止检查
        if patience_counter >= patience:
            print(f"提前停止在epoch {epoch+1}，最佳验证损失: {best_val_loss:.4f}")
            break
    
    # 加载最佳模型状态并保存
    if best_model_state:
        model.load_state_dict(best_model_state['model_state_dict'])
        torch.save(best_model_state, model_save_path)
        print(f"最佳模型已保存到 {model_save_path}")
    
    return model, history

def plot_training_history(history, save_path=None):
    """绘制训练历史"""
    plt.figure(figsize=(15, 5))
    
    # 绘制损失
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # 绘制RMSE
    plt.subplot(1, 2, 2)
    plt.plot(history['val_rmse'], label='Validation RMSE')
    plt.title('Validation RMSE')
    plt.xlabel('Epoch')
    plt.ylabel('RMSE')
    plt.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

def evaluate_model(model, test_loader, device):
    """评估模型性能"""
    model.eval()
    all_targets = []
    all_predictions = []
    
    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc="评估模型"):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # 如果有NaN，替换为0
            inputs[torch.isnan(inputs)] = 0
            
            # 前向传播
            outputs = model(inputs)
            
            # 收集预测结果
            all_targets.extend(targets.cpu().numpy())
            all_predictions.extend(outputs.squeeze().cpu().numpy())
    
    # 计算指标
    rmse = np.sqrt(mean_squared_error(all_targets, all_predictions))
    mae = mean_absolute_error(all_targets, all_predictions)
    r2 = r2_score(all_targets, all_predictions)
    
    print("\n--- 模型评估结果 ---")
    print(f"均方根误差 (RMSE): {rmse:.4f}")
    print(f"平均绝对误差 (MAE): {mae:.4f}")
    print(f"R² 分数: {r2:.4f}")
    
    return rmse, mae, r2, all_targets, all_predictions

def count_tif_files(root_dir, type_to_folder):
    """统计各类TIF文件的数量"""
    counts = {}
    for data_type, folder in type_to_folder.items():
        folder_path = os.path.join(root_dir, folder)
        if os.path.exists(folder_path):
            tif_files = glob.glob(os.path.join(folder_path, "*.tif"))
            counts[data_type] = len(tif_files)
    
    return counts

# --- 主函数 ---
if __name__ == "__main__":
    print("--- 开始使用PyTorch训练ST-ResNet模型，直接从TIF文件读取数据 ---")
    
    # 统计可用的TIF文件
    print("\n[1/7] 检查可用的TIF文件...")
    tif_counts = count_tif_files(ROOT_DIR, type_to_folder)
    for data_type, count in tif_counts.items():
        print(f"{data_type}: {count} 个TIF文件")
    
    # 加载并准备数据
    print("\n[2/7] 加载并准备标签数据...")
    train_df, test_df = load_and_prepare_data(
        LABEL_FILE, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    
    # 创建数据集和数据加载器
    print("\n[3/7] 创建训练和测试数据集...")
    train_dataset = XCO2Dataset(train_df, ROOT_DIR, use_patches=True, patch_size=3)
    test_dataset = XCO2Dataset(test_df, ROOT_DIR, use_patches=True, patch_size=3)
    
    # 获取样本形状以配置模型
    sample_features, _ = train_dataset[0]
    in_channels = sample_features.shape[0]
    print(f"样本特征形状: {sample_features.shape}")
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    # 初始化模型
    print("\n[4/7] 初始化ST-ResNet模型...")
    model = STResNet(
        in_channels=in_channels, 
        patch_size=3,
        growth_rate=32, 
        block_layers=4,
        num_dense_blocks=2, 
        num_identity_blocks=2
    ).to(device)
    
    print(model)
    
    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # 训练模型
    print("\n[5/7] 开始训练模型...")
    trained_model, history = train_model(
        model, train_loader, test_loader, criterion, optimizer, device,
        num_epochs=EPOCHS, patience=EARLY_STOP_PATIENCE, model_save_path=MODEL_SAVE_PATH
    )
    
    # 绘制训练历史
    print("\n[6/7] 绘制训练历史...")
    plot_dir = os.path.dirname(MODEL_SAVE_PATH)
    plot_path = os.path.join(plot_dir, "st_resnet_pytorch_training_history.png")
    plot_training_history(history, save_path=plot_path)
    
    # 评估模型
    print("\n[7/7] 评估最终模型性能...")
    rmse, mae, r2, targets, predictions = evaluate_model(trained_model, test_loader, device)
    
    # 绘制预测与实际值的散点图
    plt.figure(figsize=(10, 8))
    plt.scatter(targets, predictions, alpha=0.5)
    plt.plot([min(targets), max(targets)], [min(targets), max(targets)], 'r--')
    plt.xlabel('实际 XCO2')
    plt.ylabel('预测 XCO2')
    plt.title(f'预测 vs 实际 XCO2 (RMSE={rmse:.4f}, R²={r2:.4f})')
    
    scatter_path = os.path.join(plot_dir, "st_resnet_pytorch_predictions.png")
    plt.savefig(scatter_path)
    plt.show()
    
    print("\n--- 训练完成 ---") 
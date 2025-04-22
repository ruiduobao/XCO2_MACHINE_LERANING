import os
import numpy as np
import pandas as pd
import rasterio
from tqdm import tqdm
from glob import glob
from datetime import datetime
import torch
from torch.utils.data import Dataset, DataLoader

def list_xco2_files(root_dir, years_range=None, pattern="*_XGBOOST_XCO2.tif"):
    """
    列出指定目录中匹配模式的所有 XCO2 TIF 文件
    
    参数:
        root_dir (str): 包含 XCO2 TIF 文件的目录
        years_range (tuple): 可选的 (start_year, end_year) 元组用于过滤文件
        pattern (str): 匹配的文件模式
        
    返回:
        list: 排序好的文件路径列表
    """
    all_files = glob(os.path.join(root_dir, pattern))
    
    # Extract dates from filenames and sort chronologically
    dated_files = []
    for file_path in all_files:
        filename = os.path.basename(file_path)
        try:
            # Extract year and month from filename (assuming format like "2018_03_XGBOOST_XCO2.tif")
            year_month = filename.split('_')[0:2]
            year = int(year_month[0])
            month = int(year_month[1])
            
            # Filter by years range if specified
            if years_range and (year < years_range[0] or year > years_range[1]):
                continue
                
            # Create a datetime object for sorting
            file_date = datetime(year, month, 1)
            dated_files.append((file_date, file_path))
        except (ValueError, IndexError):
            print(f"Warning: Could not parse date from filename: {filename}, skipping.")
    
    # Sort by date
    dated_files.sort(key=lambda x: x[0])
    return [f[1] for f in dated_files]

def load_tif_file(file_path, normalize=True, preserve_mask=False):
    """
    将 GeoTIFF 文件加载为 numpy 数组
    
    参数:
        file_path (str): TIF 文件路径
        normalize (bool): 是否将数据归一化到 [0,1] 范围
        preserve_mask (bool): 是否返回有效数据的掩膜
        
    返回:
        tuple: 如果 preserve_mask=False: (data_array, profile)
               如果 preserve_mask=True: (data_array, valid_mask, profile)
    """
    with rasterio.open(file_path) as src:
        data = src.read(1)  # Read first band
        profile = src.profile.copy()
        
        # 创建有效数据掩膜 (True=有效, False=无效)
        valid_mask = np.ones_like(data, dtype=bool)
        
        # 标记NoData值
        if src.nodata is not None:
            data = data.astype(np.float32)
            valid_mask = data != src.nodata
            data[~valid_mask] = np.nan
            
        # 处理额外的常见NoData值
        additional_nodata_mask = data == -9999
        valid_mask = valid_mask & ~additional_nodata_mask
        data[additional_nodata_mask] = np.nan
        
        # 简单归一化（如果需要）
        if normalize and not np.all(np.isnan(data)):
            valid_data = data[~np.isnan(data)]
            if len(valid_data) > 0:
                min_val = np.min(valid_data)
                max_val = np.max(valid_data)
                if max_val > min_val:  # 确保分母不为零
                    data = np.where(~np.isnan(data), (data - min_val) / (max_val - min_val), np.nan)
                else:
                    # 如果数据无变化，则有效区域设为零，无效区域保持为NaN
                    data = np.where(~np.isnan(data), 0, np.nan)
                    
        # 如果不需要保留掩膜，则按照之前的逻辑将NaN替换为0
        if not preserve_mask:
            data = np.nan_to_num(data, nan=0.0)
            return data, profile
        else:
            # 返回数据、掩膜和配置信息
            # 将NaN值替换为0，但同时返回掩膜以便后续处理
            masked_data = np.nan_to_num(data, nan=0.0)
            return masked_data, ~np.isnan(data), profile

def create_sequence_data(file_list, sequence_length, stride=1):
    """
    从文件列表创建 ConvLSTM 训练用的序列
    
    参数:
        file_list (list): TIF 文件路径列表
        sequence_length (int): 要创建的序列长度
        stride (int): 序列之间的步长
        
    返回:
        list: 序列列表，每个序列是文件路径的列表
    """
    sequences = []
    
    for i in range(0, len(file_list) - sequence_length + 1, stride):
        sequence = file_list[i:i + sequence_length]
        sequences.append(sequence)
        
    return sequences

def load_satellite_validation_data(csv_path):
    """
    从 CSV 文件加载卫星 XCO2 验证数据
    
    参数:
        csv_path (str): 包含验证数据的 CSV 文件路径
        
    返回:
        pd.DataFrame: 包含验证数据的 DataFrame
    """
    return pd.read_csv(csv_path)

class XCO2SequenceDataset(Dataset):
    """
    XCO2 序列数据集
    
    对于每个序列:
    - X 是形状为 [sequence_length, channels, height, width] 的张量
    - y 是形状为 [1, height, width] 的下一个时间步的目标张量
    """
    def __init__(self, sequences, target_files=None, input_size=None, transform=None, mask_file=None):
        """
        Initialize the dataset.
        
        Args:
            sequences (list): List of sequences, each a list of file paths
            target_files (list): Optional list of target file paths
            input_size (tuple): Optional (height, width) to resize inputs
            transform: Optional transform to apply to loaded data
            mask_file (str): Optional path to a mask file defining valid regions
        """
        self.sequences = sequences
        self.target_files = target_files  # If None, use next file after sequence
        self.input_size = input_size
        self.transform = transform
        self.mask_file = mask_file
        
        # Cache for loaded data to improve performance
        self.cache = {}
        
        # Load mask if provided
        self.region_mask = None
        if mask_file is not None and os.path.exists(mask_file):
            with rasterio.open(mask_file) as src:
                mask_data = src.read(1)
                self.region_mask = mask_data == 1  # 值为1的区域被视为有效
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        
        # Load sequence data (X)
        sequence_data = []
        sequence_masks = []
        for file_path in sequence:
            if file_path in self.cache:
                if isinstance(self.cache[file_path], tuple) and len(self.cache[file_path]) == 2:
                    data, mask = self.cache[file_path]
                else:
                    data = self.cache[file_path]
                    mask = None
            else:
                if self.region_mask is not None:
                    data, mask, _ = load_tif_file(file_path, normalize=True, preserve_mask=True)
                    # Combine with region mask if available
                    mask = np.logical_and(mask, self.region_mask) if mask is not None else self.region_mask
                    self.cache[file_path] = (data, mask)
                else:
                    data, _ = load_tif_file(file_path, normalize=True)
                    mask = None
                    self.cache[file_path] = data
            
            sequence_data.append(data)
            if mask is not None:
                sequence_masks.append(mask)
        
        # Stack into a single array [sequence_length, height, width]
        X = np.stack(sequence_data)
        
        # Add channel dimension to get [sequence_length, channels, height, width]
        X = X[:, np.newaxis, :, :]
        
        # Create combined mask if we have masks
        combined_mask = None
        if sequence_masks:
            combined_mask = np.logical_and.reduce(sequence_masks)
        
        # Determine target (y)
        if self.target_files is not None:
            target_file = self.target_files[idx]
        else:
            # Use the file right after the sequence as target
            seq_file_list = list(sequence)
            last_file = seq_file_list[-1]
            
            # Find the index of the last file in the original file list
            all_files = glob(os.path.dirname(last_file) + "/*_XGBOOST_XCO2.tif")
            all_files.sort()
            last_idx = all_files.index(last_file)
            
            # If this is the last file, use it as target (or handle differently)
            if last_idx + 1 < len(all_files):
                target_file = all_files[last_idx + 1]
            else:
                target_file = last_file
        
        # Load target data
        if target_file in self.cache:
            if isinstance(self.cache[target_file], tuple) and len(self.cache[target_file]) == 2:
                target_data, target_mask = self.cache[target_file]
            else:
                target_data = self.cache[target_file]
                target_mask = None
        else:
            if self.region_mask is not None:
                target_data, target_mask, _ = load_tif_file(target_file, normalize=True, preserve_mask=True)
                # Combine with region mask if available
                target_mask = np.logical_and(target_mask, self.region_mask) if target_mask is not None else self.region_mask
                self.cache[target_file] = (target_data, target_mask)
            else:
                target_data, _ = load_tif_file(target_file, normalize=True)
                target_mask = None
        
        # Update combined mask with target mask
        if target_mask is not None:
            if combined_mask is not None:
                combined_mask = np.logical_and(combined_mask, target_mask)
            else:
                combined_mask = target_mask
        
        # Resize data if input_size is specified
        if self.input_size:
            from skimage.transform import resize
            
            # Resize X
            resized_X = np.zeros((X.shape[0], X.shape[1], *self.input_size), dtype=X.dtype)
            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    resized_X[i, j] = resize(X[i, j], self.input_size, 
                                              preserve_range=True, anti_aliasing=True)
            X = resized_X
            
            # Resize target
            target_data = resize(target_data, self.input_size, 
                                 preserve_range=True, anti_aliasing=True)
            
            # Resize mask if it exists
            if combined_mask is not None:
                combined_mask = resize(combined_mask.astype(np.float32), self.input_size, 
                                       preserve_range=True, anti_aliasing=False) > 0.5
        
        # Apply transforms if specified
        if self.transform:
            X = self.transform(X)
            target_data = self.transform(target_data)
        
        # Convert to torch tensors
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(target_data, dtype=torch.float32).unsqueeze(0)  # Add channel dim
        
        # Include mask in return if we have one
        if combined_mask is not None:
            mask_tensor = torch.tensor(combined_mask, dtype=torch.bool)
            return X, y, mask_tensor
        else:
            return X, y

def get_dataloaders(root_dir, sequence_length, batch_size=8, val_split=0.2, input_size=None,
                    years_range=None, pattern="*_XGBOOST_XCO2.tif", mask_file=None):
    """
    创建训练和验证的数据加载器
    
    参数:
        root_dir (str): 包含 XCO2 TIF 文件的目录
        sequence_length (int): 要创建的序列长度
        batch_size (int): 数据加载器的批量大小
        val_split (float): 用于验证的数据比例
        input_size (tuple): 可选的 (height, width) 用于调整输入大小
        years_range (tuple): 可选的 (start_year, end_year) 元组用于过滤文件
        pattern (str): 匹配的文件模式
        mask_file (str): 可选的掩膜文件路径，用于定义有效区域
        
    返回:
        tuple: (train_loader, val_loader)
    """
    # List and sort files
    file_list = list_xco2_files(root_dir, years_range, pattern)
    
    if len(file_list) < sequence_length + 1:
        raise ValueError(f"Not enough files found. Need at least {sequence_length + 1}, but got {len(file_list)}")
    
    # Create sequences
    sequences = create_sequence_data(file_list, sequence_length)
    
    # Split into train and validation
    val_size = int(len(sequences) * val_split)
    train_sequences = sequences[:-val_size] if val_size > 0 else sequences
    val_sequences = sequences[-val_size:] if val_size > 0 else []
    
    # Create datasets
    train_dataset = XCO2SequenceDataset(train_sequences, input_size=input_size, mask_file=mask_file)
    val_dataset = XCO2SequenceDataset(val_sequences, input_size=input_size, mask_file=mask_file) if val_sequences else None
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=2, pin_memory=True) if val_dataset else None
    
    return train_loader, val_loader

def create_auxiliary_input(xco2_file, auxiliary_dirs, auxiliary_patterns=None):
    """
    创建辅助输入特征以补充 XCO2 数据
    
    参数:
        xco2_file (str): XCO2 TIF 文件路径
        auxiliary_dirs (dict): 特征名称到目录的映射字典
        auxiliary_patterns (dict): 可选的特征名称到文件模式的映射字典
        
    返回:
        numpy.ndarray: 形状为 [channels, height, width] 的数组，包含辅助数据
    """
    # Initialize auxiliary_patterns to empty dict if None
    if auxiliary_patterns is None:
        auxiliary_patterns = {}
        
    # Extract year and month from XCO2 filename
    filename = os.path.basename(xco2_file)
    try:
        year_month = filename.split('_')[0:2]
        year = int(year_month[0])
        month = int(year_month[1])
    except (ValueError, IndexError):
        raise ValueError(f"Could not parse year and month from filename: {filename}")
    
    # List to store auxiliary data arrays
    aux_data_list = []
    
    # Load XCO2 data first to get dimensions
    xco2_data, xco2_profile = load_tif_file(xco2_file, normalize=True)
    height, width = xco2_data.shape
    
    # Process each auxiliary feature
    for feature_name, feature_dir in auxiliary_dirs.items():
        # Determine file pattern
        pattern = auxiliary_patterns.get(feature_name, f"{feature_name}_{year}_{month:02d}.tif")
        
        # For annual features, use only year
        if feature_name in ['DEM', 'slope', 'aspect', 'Lantitude', 'Longtitude', 
                           'landscan', 'humanfootprint', 'CLCD', 'MODISLANDCOVER']:
            pattern = f"{feature_name}_{year}.tif"
        
        # Find matching files
        feature_files = glob(os.path.join(feature_dir, pattern))
        
        if not feature_files:
            print(f"Warning: No files found for {feature_name} with pattern {pattern}")
            # Create a dummy array of zeros with same shape as XCO2
            dummy_data = np.zeros((1, height, width), dtype=np.float32)
            aux_data_list.append(dummy_data)
            continue
        
        # Load the first matching file
        feature_file = feature_files[0]
        try:
            feature_data, _ = load_tif_file(feature_file, normalize=True)
            
            # Ensure data has same shape as XCO2
            if feature_data.shape != (height, width):
                from skimage.transform import resize
                feature_data = resize(feature_data, (height, width), 
                                      preserve_range=True, anti_aliasing=True)
            
            # Add channel dimension
            feature_data = feature_data[np.newaxis, :, :]
            aux_data_list.append(feature_data)
            
        except Exception as e:
            print(f"Error loading {feature_name}: {e}")
            # Create a dummy array of zeros
            dummy_data = np.zeros((1, height, width), dtype=np.float32)
            aux_data_list.append(dummy_data)
    
    # Stack all auxiliary data along channel dimension
    if aux_data_list:
        auxiliary_data = np.concatenate(aux_data_list, axis=0)
    else:
        auxiliary_data = np.zeros((0, height, width), dtype=np.float32)
    
    return auxiliary_data

class XCO2WithAuxDataset(Dataset):
    """
    带有辅助数据的 XCO2 序列数据集
    
    类似于 XCO2SequenceDataset，但加入了辅助特征
    """
    def __init__(self, sequences, auxiliary_dirs, target_files=None, input_size=None, transform=None, mask_file=None):
        """
        Initialize the dataset.
        
        Args:
            sequences (list): List of sequences, each a list of file paths
            auxiliary_dirs (dict): Dictionary mapping feature names to directories
            target_files (list): Optional list of target file paths
            input_size (tuple): Optional (height, width) to resize inputs
            transform: Optional transform to apply to loaded data
            mask_file (str): Optional path to a mask file defining valid regions
        """
        self.sequences = sequences
        self.auxiliary_dirs = auxiliary_dirs
        self.target_files = target_files
        self.input_size = input_size
        self.transform = transform
        self.mask_file = mask_file
        
        # Cache for loaded data
        self.cache = {}
        
        # Load mask if provided
        self.region_mask = None
        if mask_file is not None and os.path.exists(mask_file):
            with rasterio.open(mask_file) as src:
                mask_data = src.read(1)
                self.region_mask = mask_data == 1  # 值为1的区域被视为有效
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        
        # Load sequence data with auxiliary features
        sequence_data = []
        sequence_masks = []
        for file_path in sequence:
            cache_key = file_path
            if cache_key in self.cache:
                if isinstance(self.cache[cache_key], tuple) and len(self.cache[cache_key]) == 2:
                    combined_data, mask = self.cache[cache_key]
                else:
                    combined_data = self.cache[cache_key]
                    mask = None
            else:
                # Load XCO2 data with mask if region mask is provided
                if self.region_mask is not None:
                    xco2_data, xco2_mask, _ = load_tif_file(file_path, normalize=True, preserve_mask=True)
                    # Combine with region mask
                    mask = np.logical_and(xco2_mask, self.region_mask) if xco2_mask is not None else self.region_mask
                else:
                    xco2_data, _ = load_tif_file(file_path, normalize=True)
                    mask = None
                
                # Load auxiliary data
                aux_data = create_auxiliary_input(file_path, self.auxiliary_dirs)
                
                # Combine XCO2 and auxiliary data
                xco2_data = xco2_data[np.newaxis, :, :]  # Add channel dimension
                combined_data = np.concatenate([xco2_data, aux_data], axis=0)
                
                if mask is not None:
                    self.cache[cache_key] = (combined_data, mask)
                else:
                    self.cache[cache_key] = combined_data
            
            sequence_data.append(combined_data)
            if mask is not None:
                sequence_masks.append(mask)
        
        # Stack into a single array [sequence_length, channels, height, width]
        X = np.stack(sequence_data)
        
        # Create combined mask if we have masks
        combined_mask = None
        if sequence_masks:
            combined_mask = np.logical_and.reduce(sequence_masks)
        
        # Determine target (y)
        if self.target_files is not None:
            target_file = self.target_files[idx]
        else:
            # Use the file right after the sequence as target
            seq_file_list = list(sequence)
            last_file = seq_file_list[-1]
            
            # Find the index of the last file in the original file list
            all_files = glob(os.path.dirname(last_file) + "/*_XGBOOST_XCO2.tif")
            all_files.sort()
            last_idx = all_files.index(last_file)
            
            # If this is the last file, use it as target (or handle differently)
            if last_idx + 1 < len(all_files):
                target_file = all_files[last_idx + 1]
            else:
                target_file = last_file
        
        # Load target data
        if target_file in self.cache and (isinstance(self.cache[target_file], tuple) or self.cache[target_file].ndim == 3):
            # If we have cached the combined data
            if isinstance(self.cache[target_file], tuple) and len(self.cache[target_file]) == 2:
                combined_target, target_mask = self.cache[target_file]
                if combined_target.ndim == 3:  # If it's the full auxiliary data
                    target_data = combined_target[0]  # Extract just the XCO2 channel
                else:
                    target_data = combined_target
            else:
                combined_target = self.cache[target_file]
                target_data = combined_target[0] if combined_target.ndim == 3 else combined_target
                target_mask = None
        else:
            # Load target data with mask if region mask is provided
            if self.region_mask is not None:
                target_data, target_mask, _ = load_tif_file(target_file, normalize=True, preserve_mask=True)
                # Combine with region mask
                target_mask = np.logical_and(target_mask, self.region_mask) if target_mask is not None else self.region_mask
            else:
                target_data, _ = load_tif_file(target_file, normalize=True)
                target_mask = None
        
        # Update combined mask with target mask
        if target_mask is not None:
            if combined_mask is not None:
                combined_mask = np.logical_and(combined_mask, target_mask)
            else:
                combined_mask = target_mask
        
        # Resize data if input_size is specified
        if self.input_size:
            from skimage.transform import resize
            
            # Resize X
            resized_X = np.zeros((X.shape[0], X.shape[1], *self.input_size), dtype=X.dtype)
            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    resized_X[i, j] = resize(X[i, j], self.input_size, 
                                            preserve_range=True, anti_aliasing=True)
            X = resized_X
            
            # Resize target
            target_data = resize(target_data, self.input_size, 
                                preserve_range=True, anti_aliasing=True)
            
            # Resize mask if it exists
            if combined_mask is not None:
                combined_mask = resize(combined_mask.astype(np.float32), self.input_size, 
                                       preserve_range=True, anti_aliasing=False) > 0.5
        
        # Apply transforms if specified
        if self.transform:
            X = self.transform(X)
            target_data = self.transform(target_data)
        
        # Convert to torch tensors
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(target_data, dtype=torch.float32).unsqueeze(0)  # Add channel dim
        
        # Include mask in return if we have one
        if combined_mask is not None:
            mask_tensor = torch.tensor(combined_mask, dtype=torch.bool)
            return X, y, mask_tensor
        else:
            return X, y 
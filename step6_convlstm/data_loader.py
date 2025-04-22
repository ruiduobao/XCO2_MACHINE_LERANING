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
    List all XCO2 TIF files from the specified directory that match the pattern.
    
    Args:
        root_dir (str): Directory containing XCO2 TIF files
        years_range (tuple): Optional tuple of (start_year, end_year) to filter files
        pattern (str): File pattern to match
        
    Returns:
        list: Sorted list of file paths
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

def load_tif_file(file_path, normalize=True):
    """
    Load a GeoTIFF file as a numpy array.
    
    Args:
        file_path (str): Path to TIF file
        normalize (bool): Whether to normalize data to [0,1] range
        
    Returns:
        tuple: (data_array, profile) where profile contains metadata
    """
    with rasterio.open(file_path) as src:
        data = src.read(1)  # Read first band
        profile = src.profile.copy()
        
        # Replace NoData values with NaN
        if src.nodata is not None:
            data = data.astype(np.float32)
            data[data == src.nodata] = np.nan
            
        # Handle any additional common NoData values
        data[data == -9999] = np.nan
        
        # Simple normalization if requested
        if normalize and not np.all(np.isnan(data)):
            # Compute stats ignoring NaN values
            valid_data = data[~np.isnan(data)]
            if len(valid_data) > 0:
                min_val = np.nanmin(data)
                max_val = np.nanmax(data)
                if max_val > min_val:
                    data = (data - min_val) / (max_val - min_val)
                    
        # Replace NaN values with 0 after normalization
        data = np.nan_to_num(data, nan=0.0)
            
    return data, profile

def create_sequence_data(file_list, sequence_length, stride=1):
    """
    Create sequences from list of files for ConvLSTM training.
    
    Args:
        file_list (list): List of TIF file paths
        sequence_length (int): Length of sequences to create
        stride (int): Stride between sequences
        
    Returns:
        list: List of sequences where each sequence is a list of file paths
    """
    sequences = []
    
    for i in range(0, len(file_list) - sequence_length + 1, stride):
        sequence = file_list[i:i + sequence_length]
        sequences.append(sequence)
        
    return sequences

def load_satellite_validation_data(csv_path):
    """
    Load satellite XCO2 validation data from CSV file.
    
    Args:
        csv_path (str): Path to CSV file with validation data
        
    Returns:
        pd.DataFrame: DataFrame containing validation data
    """
    return pd.read_csv(csv_path)

class XCO2SequenceDataset(Dataset):
    """
    Dataset for XCO2 sequences.
    
    For each sequence:
    - X is a tensor of shape [sequence_length, channels, height, width]
    - y is the target tensor of shape [1, height, width] for the next time step
    """
    def __init__(self, sequences, target_files=None, input_size=None, transform=None):
        """
        Initialize the dataset.
        
        Args:
            sequences (list): List of sequences, each a list of file paths
            target_files (list): Optional list of target file paths
            input_size (tuple): Optional (height, width) to resize inputs
            transform: Optional transform to apply to loaded data
        """
        self.sequences = sequences
        self.target_files = target_files  # If None, use next file after sequence
        self.input_size = input_size
        self.transform = transform
        
        # Cache for loaded data to improve performance
        self.cache = {}
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        
        # Load sequence data (X)
        sequence_data = []
        for file_path in sequence:
            if file_path in self.cache:
                data = self.cache[file_path]
            else:
                data, _ = load_tif_file(file_path)
                self.cache[file_path] = data
            sequence_data.append(data)
        
        # Stack into a single array [sequence_length, height, width]
        X = np.stack(sequence_data)
        
        # Add channel dimension to get [sequence_length, channels, height, width]
        X = X[:, np.newaxis, :, :]
        
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
            target_data = self.cache[target_file]
        else:
            target_data, _ = load_tif_file(target_file)
            self.cache[target_file] = target_data
        
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
        
        # Apply transforms if specified
        if self.transform:
            X = self.transform(X)
            target_data = self.transform(target_data)
        
        # Convert to torch tensors
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(target_data, dtype=torch.float32).unsqueeze(0)  # Add channel dim
        
        return X, y

def get_dataloaders(root_dir, sequence_length, batch_size=8, val_split=0.2, input_size=None,
                    years_range=None, pattern="*_XGBOOST_XCO2.tif"):
    """
    Create dataloaders for training and validation.
    
    Args:
        root_dir (str): Directory containing XCO2 TIF files
        sequence_length (int): Length of sequences to create
        batch_size (int): Batch size for dataloaders
        val_split (float): Fraction of data to use for validation
        input_size (tuple): Optional (height, width) to resize inputs
        years_range (tuple): Optional tuple of (start_year, end_year) to filter files
        pattern (str): File pattern to match
        
    Returns:
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
    train_dataset = XCO2SequenceDataset(train_sequences, input_size=input_size)
    val_dataset = XCO2SequenceDataset(val_sequences, input_size=input_size) if val_sequences else None
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size) if val_dataset else None
    
    return train_loader, val_loader

def create_auxiliary_input(xco2_file, auxiliary_dirs, auxiliary_patterns=None):
    """
    Create auxiliary input features to complement XCO2 data.
    
    Args:
        xco2_file (str): Path to XCO2 TIF file
        auxiliary_dirs (dict): Dictionary mapping feature names to directories
        auxiliary_patterns (dict): Optional dictionary mapping feature names to file patterns
        
    Returns:
        numpy.ndarray: Array with shape [channels, height, width] containing auxiliary data
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
    Dataset for XCO2 sequences with auxiliary data.
    """
    def __init__(self, sequences, auxiliary_dirs, target_files=None, input_size=None, transform=None):
        """
        Initialize the dataset.
        
        Args:
            sequences (list): List of sequences, each a list of file paths
            auxiliary_dirs (dict): Dictionary mapping feature names to directories
            target_files (list): Optional list of target file paths
            input_size (tuple): Optional (height, width) to resize inputs
            transform: Optional transform to apply to loaded data
        """
        self.sequences = sequences
        self.auxiliary_dirs = auxiliary_dirs
        self.target_files = target_files
        self.input_size = input_size
        self.transform = transform
        
        # Cache for loaded data
        self.cache = {}
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        
        # Load sequence data with auxiliary features
        sequence_data = []
        for file_path in sequence:
            cache_key = file_path
            if cache_key in self.cache:
                combined_data = self.cache[cache_key]
            else:
                # Load XCO2 data
                xco2_data, _ = load_tif_file(file_path)
                
                # Load auxiliary data
                aux_data = create_auxiliary_input(file_path, self.auxiliary_dirs)
                
                # Combine XCO2 and auxiliary data
                xco2_data = xco2_data[np.newaxis, :, :]  # Add channel dimension
                combined_data = np.concatenate([xco2_data, aux_data], axis=0)
                
                self.cache[cache_key] = combined_data
            
            sequence_data.append(combined_data)
        
        # Stack into a single array [sequence_length, channels, height, width]
        X = np.stack(sequence_data)
        
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
        if target_file in self.cache and isinstance(self.cache[target_file], np.ndarray) and self.cache[target_file].ndim == 3:
            # If we have cached the combined data, extract just the XCO2 channel
            target_data = self.cache[target_file][0]
        else:
            target_data, _ = load_tif_file(target_file)
        
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
        
        # Apply transforms if specified
        if self.transform:
            X = self.transform(X)
            target_data = self.transform(target_data)
        
        # Convert to torch tensors
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(target_data, dtype=torch.float32).unsqueeze(0)  # Add channel dim
        
        return X, y 
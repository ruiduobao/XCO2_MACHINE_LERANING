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
from step6_convlstm.data_loader import list_xco2_files, load_tif_file, create_auxiliary_input
from step6_convlstm.model import XCO2ConvLSTM, predict_next_step, predict_multiple_steps


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Predict XCO2 using trained ConvLSTM model')
    
    # Model parameters
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model weights (.pth file)')
    parser.add_argument('--config_path', type=str,
                        help='Path to model configuration file (optional)')
    
    # Input parameters
    parser.add_argument('--data_dir', type=str, default=r'E:\地理所\论文\中国XCO2论文_2025.04\数据\XGBOOST_XCO2',
                        help='Directory containing XCO2 TIF files')
    parser.add_argument('--sequence_length', type=int, default=6,
                        help='Length of input sequences (in months)')
    parser.add_argument('--input_size', type=int, nargs=2, default=None,
                        help='Optional size to resize inputs, e.g., --input_size 128 128')
    parser.add_argument('--aux_data', action='store_true',
                        help='Whether to use auxiliary data features')
    
    # Prediction parameters
    parser.add_argument('--start_date', type=str, default=None,
                        help='Start date for prediction sequence (YYYY-MM format)')
    parser.add_argument('--num_steps', type=int, default=12,
                        help='Number of steps (months) to predict')
    
    # Output parameters
    parser.add_argument('--output_dir', type=str, default='./predictions',
                        help='Directory to save prediction TIFs')
    parser.add_argument('--visualize', action='store_true',
                        help='Generate visualizations of predictions')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU index to use (-1 for CPU)')
    
    return parser.parse_args()


def setup_auxiliary_data_dirs():
    """
    Setup directories for auxiliary data.
    
    Returns:
        dict: Dictionary mapping feature names to directories
    """
    root_dir = r'E:\地理所\论文\中国XCO2论文_2025.04\数据'
    
    # Map feature names to directories (same as in train.py)
    aux_dirs = {
        'Lantitude': os.path.join(root_dir, '纬度栅格'),
        'Longtitude': os.path.join(root_dir, '经度栅格'),
        # 'UnixTime': os.path.join(root_dir, '每月时间戳的栅格数据'),
        # 'aspect': os.path.join(root_dir, '坡向数据'),
        # 'slope': os.path.join(root_dir, '坡度数据'),
        # 'DEM': os.path.join(root_dir, 'DEM'),
        # 'VIIRS': os.path.join(root_dir, '夜光遥感'),
        'ERA5Land': os.path.join(root_dir, 'ERA5'),
        # 'AOD': os.path.join(root_dir, '气溶胶厚度'),
        'CT2019B': os.path.join(root_dir, 'carbon_tracer'),
        'landscan': os.path.join(root_dir, 'landscan'),
        # 'odiac1km': os.path.join(root_dir, 'odiac'),
        # 'humanfootprint': os.path.join(root_dir, '人类足迹数据'),
        'OCO2GEOS': os.path.join(root_dir, 'OCO2_GEOS_XCO2同化数据'),
        'CAMStcco2': os.path.join(root_dir, 'CAMS'),
        # 'CLCD': os.path.join(root_dir, 'CLCD'),
        'MODISLANDCOVER': os.path.join(root_dir, 'modis_landcover'),
        # 'MOD13A2': os.path.join(root_dir, 'NDVI')
    }
    
    return aux_dirs


def create_prediction_sequence(xco2_files, start_date, sequence_length, aux_dirs=None, input_size=None):
    """
    Create a sequence tensor for prediction starting from a specific date.
    
    Args:
        xco2_files (list): List of XCO2 TIF files
        start_date (str): Start date in 'YYYY-MM' format for the end of the sequence
        sequence_length (int): Length of input sequence
        aux_dirs (dict): Dictionary of auxiliary data directories (optional)
        input_size (tuple): Optional size to resize inputs
        
    Returns:
        tuple: (sequence_tensor, tif_profile) for prediction and saving output
    """
    # Parse start date
    try:
        year, month = map(int, start_date.split('-'))
        target_date = datetime(year, month, 1)
    except (ValueError, AttributeError):
        raise ValueError(f"Invalid start_date format: {start_date}. Use YYYY-MM format.")
    
    # Find files to include in sequence
    sequence_files = []
    sorted_files = sorted(xco2_files, key=lambda x: os.path.basename(x))
    
    # Collect files before the target date
    for file_path in sorted_files:
        filename = os.path.basename(file_path)
        try:
            file_year, file_month = map(int, filename.split('_')[:2])
            file_date = datetime(file_year, file_month, 1)
            
            # Check if file date is before or equal to target date
            if file_date <= target_date:
                sequence_files.append(file_path)
            else:
                # Skip files after the target date
                break
        except (ValueError, IndexError):
            print(f"Warning: Skipping file with invalid filename format: {filename}")
    
    # Take the last N files to form the sequence
    if len(sequence_files) < sequence_length:
        raise ValueError(f"Not enough files before {start_date} to create a sequence of length {sequence_length}")
    
    sequence_files = sequence_files[-sequence_length:]
    
    # Load sequence data
    sequence_data = []
    tif_profile = None
    
    for file_path in sequence_files:
        if aux_dirs is not None:
            # With auxiliary data
            xco2_data, profile = load_tif_file(file_path, normalize=True)
            if tif_profile is None:
                tif_profile = profile
                
            # Load auxiliary data
            aux_data = create_auxiliary_input(file_path, aux_dirs)
            
            # Combine XCO2 and auxiliary data
            xco2_data = xco2_data[np.newaxis, :, :]  # Add channel dimension
            combined_data = np.concatenate([xco2_data, aux_data], axis=0)
            sequence_data.append(combined_data)
        else:
            # XCO2 only
            data, profile = load_tif_file(file_path, normalize=True)
            if tif_profile is None:
                tif_profile = profile
            sequence_data.append(data)
    
    # Stack into array
    if aux_dirs is None:
        # For XCO2 only, add channel dimension
        sequence_array = np.stack(sequence_data)
        sequence_array = sequence_array[:, np.newaxis, :, :]  # [seq_len, channels, height, width]
    else:
        # With auxiliary data, already has channel dimension
        sequence_array = np.stack(sequence_data)  # [seq_len, channels, height, width]
    
    # Resize if needed
    if input_size is not None:
        from skimage.transform import resize
        resized_array = np.zeros((sequence_array.shape[0], sequence_array.shape[1], *input_size), dtype=sequence_array.dtype)
        for i in range(sequence_array.shape[0]):
            for j in range(sequence_array.shape[1]):
                resized_array[i, j] = resize(sequence_array[i, j], input_size, 
                                             preserve_range=True, anti_aliasing=True)
        sequence_array = resized_array
    
    # Convert to torch tensor and add batch dimension
    sequence_tensor = torch.tensor(sequence_array, dtype=torch.float32).unsqueeze(0)
    
    return sequence_tensor, tif_profile


def denormalize_prediction(prediction, reference_file):
    """
    Denormalize predicted values to original range.
    
    Args:
        prediction (numpy.ndarray): Normalized prediction array
        reference_file (str): Reference file path to get original value range
        
    Returns:
        numpy.ndarray: Denormalized prediction array
    """
    with rasterio.open(reference_file) as src:
        # Read data and find min/max values
        reference_data = src.read(1)
        nodata_value = src.nodata
        
        # Mask NoData values
        if nodata_value is not None:
            mask = reference_data != nodata_value
            if np.any(mask):
                min_val = np.min(reference_data[mask])
                max_val = np.max(reference_data[mask])
            else:
                min_val, max_val = 0, 1  # Default if all NoData
        else:
            min_val = np.min(reference_data)
            max_val = np.max(reference_data)
    
    # Apply denormalization: scaled_x = x * (max - min) + min
    denormalized = prediction * (max_val - min_val) + min_val
    return denormalized


def save_prediction_as_tif(prediction, profile, output_path):
    """
    Save prediction array as a GeoTIFF file.
    
    Args:
        prediction (numpy.ndarray): 2D prediction array
        profile (dict): Profile with georeference information
        output_path (str): Path to save the output TIF file
    """
    # Update profile for output
    profile.update({
        'count': 1,
        'dtype': 'float32',
        'compress': 'lzw',
        'tiled': True,
        'blockxsize': 256,
        'blockysize': 256
    })
    
    # Write to file
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(prediction.astype(np.float32), 1)


def visualize_predictions(predictions, output_dir, start_date):
    """
    Generate visualizations of the prediction results.
    
    Args:
        predictions (list): List of prediction arrays
        output_dir (str): Directory to save visualizations
        start_date (str): Start date in 'YYYY-MM' format
    """
    # Create visualization directory
    viz_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(viz_dir, exist_ok=True)
    
    # Parse start date
    year, month = map(int, start_date.split('-'))
    prediction_date = datetime(year, month, 1)
    
    # Create visualizations
    for i, prediction in enumerate(predictions):
        # Calculate the date for this prediction
        if i > 0:  # Skip the first prediction (which is the start date)
            prediction_date = prediction_date + relativedelta(months=1)
        
        pred_date_str = prediction_date.strftime('%Y-%m')
        
        # Create figure
        plt.figure(figsize=(10, 8))
        plt.imshow(prediction, cmap='viridis')
        plt.colorbar(label='XCO2')
        plt.title(f'Predicted XCO2 for {pred_date_str}')
        plt.tight_layout()
        
        # Save figure
        filename = f'prediction_{pred_date_str}.png'
        plt.savefig(os.path.join(viz_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()


def main():
    """Main prediction script."""
    # Parse arguments
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
        print(f"Using GPU: {torch.cuda.get_device_name(args.gpu)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    # Load model configuration
    if args.config_path and os.path.exists(args.config_path):
        import json
        with open(args.config_path, 'r') as f:
            config = json.load(f)
        print(f"Loaded model configuration from {args.config_path}")
        
        # Override some arguments with values from config
        hidden_dims = config.get('hidden_dims', [32, 64])
        num_layers = len(hidden_dims)
        input_channels = 1 + (18 if args.aux_data else 0)  # Assume 18 auxiliary features if aux_data is True
        kernel_size = config.get('kernel_size', 3)
        dropout = config.get('dropout', 0.2)
    else:
        # Default configuration
        hidden_dims = [32, 64]
        num_layers = len(hidden_dims)
        input_channels = 1 + (18 if args.aux_data else 0)
        kernel_size = 3
        dropout = 0.2
        print("Using default model configuration")
    
    # Initialize model
    model = XCO2ConvLSTM(
        input_channels=input_channels,
        hidden_dims=hidden_dims,
        kernel_size=kernel_size,
        num_layers=num_layers,
        dropout=dropout
    )
    
    # Load model weights
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model = model.to(device)
    model.eval()
    print(f"Model loaded from {args.model_path}")
    
    # Set up auxiliary data directories if needed
    aux_dirs = setup_auxiliary_data_dirs() if args.aux_data else None
    
    # List and sort XCO2 files
    xco2_files = list_xco2_files(args.data_dir)
    
    # Determine start date if not provided
    if args.start_date is None:
        # Use the date of the last available file
        last_file = xco2_files[-1]
        filename = os.path.basename(last_file)
        try:
            year, month = map(int, filename.split('_')[:2])
            args.start_date = f"{year}-{month:02d}"
        except (ValueError, IndexError):
            print("Error: Could not determine start date from last file")
            return
    
    print(f"Prediction start date: {args.start_date}")
    
    # Create input sequence
    try:
        sequence_tensor, tif_profile = create_prediction_sequence(
            xco2_files, args.start_date, args.sequence_length, aux_dirs, args.input_size
        )
        print(f"Created input sequence of shape {sequence_tensor.shape}")
    except Exception as e:
        print(f"Error creating input sequence: {e}")
        return
    
    # Generate predictions
    predictions = predict_multiple_steps(model, sequence_tensor, args.num_steps, device)
    print(f"Generated {len(predictions)} predictions")
    
    # Parse start date for naming output files
    year, month = map(int, args.start_date.split('-'))
    prediction_date = datetime(year, month, 1)
    
    # Reference file for denormalization (use the last file in the sequence)
    reference_file = xco2_files[-1]
    
    # Save predictions as TIF files
    print("Saving prediction TIFs...")
    for i, prediction in enumerate(tqdm(predictions)):
        # Move forward one month for each prediction
        prediction_date = prediction_date + relativedelta(months=1)
        pred_date_str = prediction_date.strftime('%Y_%m')
        
        # Denormalize prediction
        denormalized = denormalize_prediction(prediction, reference_file)
        
        # Save as TIF
        output_path = os.path.join(args.output_dir, f"{pred_date_str}_CONVLSTM_XCO2.tif")
        save_prediction_as_tif(denormalized, tif_profile, output_path)
    
    print(f"Saved {len(predictions)} prediction TIFs to {args.output_dir}")
    
    # Generate visualizations if requested
    if args.visualize:
        print("Generating visualizations...")
        visualize_predictions(predictions, args.output_dir, args.start_date)
        print(f"Visualizations saved to {os.path.join(args.output_dir, 'visualizations')}")
    
    print("Prediction complete!")


if __name__ == "__main__":
    main() 
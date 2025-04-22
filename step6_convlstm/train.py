import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our modules
from step6_convlstm.data_loader import get_dataloaders, XCO2WithAuxDataset, create_sequence_data, list_xco2_files
from step6_convlstm.model import XCO2ConvLSTM, train_model, predict_next_step, EarlyStopping

def parse_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description='Train ConvLSTM model for XCO2 prediction')
    
    # Data parameters
    parser.add_argument('--data_dir', type=str, default=r'E:\地理所\论文\中国XCO2论文_2025.04\数据\XGBOOST_XCO2',
                        help='Directory containing XCO2 TIF files')
    parser.add_argument('--aux_data', action='store_true',
                        help='Whether to use auxiliary data features')
    parser.add_argument('--input_size', type=int, nargs=2, default=None,
                        help='Optional size to resize inputs, e.g., --input_size 128 128')
    parser.add_argument('--sequence_length', type=int, default=6,
                        help='Length of input sequences (in months)')
    parser.add_argument('--years_range', type=int, nargs=2, default=[2015, 2021],
                        help='Range of years to use for training, e.g., --years_range 2015 2021')
    
    # Model parameters
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=[32, 64],
                        help='Hidden dimensions for each ConvLSTM layer')
    parser.add_argument('--kernel_size', type=int, default=3,
                        help='Kernel size for ConvLSTM layers')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='Dropout probability')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs to train for')
    parser.add_argument('--patience', type=int, default=10,
                        help='Patience for early stopping')
    parser.add_argument('--val_split', type=float, default=0.2,
                        help='Fraction of data to use for validation')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU index to use (-1 for CPU)')
    
    # Output parameters
    parser.add_argument('--output_dir', type=str, default='./results',
                        help='Directory to save results')
    parser.add_argument('--experiment_name', type=str, default=None,
                        help='Name of experiment for saving results')
    
    return parser.parse_args()

def setup_auxiliary_data_dirs():
    """
    Setup directories for auxiliary data.
    
    Returns:
        dict: Dictionary mapping feature names to directories
    """
    root_dir = r'E:\地理所\论文\中国XCO2论文_2025.04\数据'
    
    # Map feature names to directories
    aux_dirs = {
        'Lantitude': os.path.join(root_dir, '纬度栅格'),
        'Longtitude': os.path.join(root_dir, '经度栅格'),
        'UnixTime': os.path.join(root_dir, '每月时间戳的栅格数据'),
        'aspect': os.path.join(root_dir, '坡向数据'),
        'slope': os.path.join(root_dir, '坡度数据'),
        'DEM': os.path.join(root_dir, 'DEM'),
        'VIIRS': os.path.join(root_dir, '夜光遥感'),
        'ERA5Land': os.path.join(root_dir, 'ERA5'),
        'AOD': os.path.join(root_dir, '气溶胶厚度'),
        'CT2019B': os.path.join(root_dir, 'carbon_tracer'),
        'landscan': os.path.join(root_dir, 'landscan'),
        'odiac1km': os.path.join(root_dir, 'odiac'),
        'humanfootprint': os.path.join(root_dir, '人类足迹数据'),
        'OCO2GEOS': os.path.join(root_dir, 'OCO2_GEOS_XCO2同化数据'),
        'CAMStcco2': os.path.join(root_dir, 'CAMS'),
        'CLCD': os.path.join(root_dir, 'CLCD'),
        'MODISLANDCOVER': os.path.join(root_dir, 'modis_landcover'),
        'MOD13A2': os.path.join(root_dir, 'NDVI')
    }
    
    return aux_dirs

def visualize_prediction(model, val_loader, device, output_dir, experiment_name=None):
    """
    Generate and save sample predictions for visual inspection.
    
    Args:
        model (nn.Module): Trained model
        val_loader (DataLoader): Validation data loader
        device: Device to run prediction on
        output_dir (str): Directory to save visualizations
        experiment_name (str): Optional name for the output file
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get a sample batch from the validation loader
    inputs, targets = next(iter(val_loader))
    
    # Select a random sample from the batch
    sample_idx = np.random.randint(inputs.shape[0])
    sample_input = inputs[sample_idx:sample_idx+1].to(device)  # Add batch dimension back
    sample_target = targets[sample_idx].squeeze().cpu().numpy()
    
    # Generate prediction
    with torch.no_grad():
        sample_output = model(sample_input).squeeze().cpu().numpy()
    
    # Visualize the last input, target, and prediction
    plt.figure(figsize=(15, 5))
    
    # Plot last input frame
    plt.subplot(131)
    plt.title('Last Input Frame')
    plt.imshow(sample_input[0, -1, 0].cpu().numpy(), cmap='viridis')
    plt.colorbar(label='XCO2')
    
    # Plot target
    plt.subplot(132)
    plt.title('Target')
    plt.imshow(sample_target, cmap='viridis')
    plt.colorbar(label='XCO2')
    
    # Plot prediction
    plt.subplot(133)
    plt.title('Prediction')
    plt.imshow(sample_output, cmap='viridis')
    plt.colorbar(label='XCO2')
    
    # Save figure
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    filename = f'prediction_{experiment_name}_{timestamp}.png' if experiment_name else f'prediction_{timestamp}.png'
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visualization saved to {os.path.join(output_dir, filename)}")

def plot_training_history(history, output_dir, experiment_name=None):
    """
    Plot training and validation loss curves.
    
    Args:
        history (dict): Training history dictionary
        output_dir (str): Directory to save the plot
        experiment_name (str): Optional name for the output file
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    # Mark the best epoch
    best_epoch = history['best_epoch']
    best_val_loss = history['best_val_loss']
    plt.scatter(best_epoch, best_val_loss, color='red', s=100, zorder=5)
    plt.annotate(f'Best: {best_val_loss:.4f}',
                 (best_epoch, best_val_loss),
                 xytext=(10, -20),
                 textcoords='offset points',
                 arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'))
    
    # Save figure
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    filename = f'training_history_{experiment_name}_{timestamp}.png' if experiment_name else f'training_history_{timestamp}.png'
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Training history plot saved to {os.path.join(output_dir, filename)}")

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Set experiment name if not provided
    if args.experiment_name is None:
        args.experiment_name = f"convlstm_seq{args.sequence_length}_aux{args.aux_data}_dims{'-'.join(map(str, args.hidden_dims))}"
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
        print(f"Using GPU: {torch.cuda.get_device_name(args.gpu)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    # Set up auxiliary data directories if needed
    aux_dirs = setup_auxiliary_data_dirs() if args.aux_data else None
    
    # List and sort XCO2 files
    xco2_files = list_xco2_files(args.data_dir, args.years_range)
    
    if len(xco2_files) < args.sequence_length + 1:
        print(f"Error: Not enough files found. Need at least {args.sequence_length + 1}, but got {len(xco2_files)}")
        return
    
    print(f"Found {len(xco2_files)} XCO2 files")
    
    # Create sequences
    sequences = create_sequence_data(xco2_files, args.sequence_length)
    
    # Split into train and validation
    val_size = int(len(sequences) * args.val_split)
    train_sequences = sequences[:-val_size] if val_size > 0 else sequences
    val_sequences = sequences[-val_size:] if val_size > 0 else []
    
    print(f"Training sequences: {len(train_sequences)}")
    print(f"Validation sequences: {len(val_sequences)}")
    
    # Create datasets and dataloaders
    if args.aux_data:
        print("Using auxiliary data features")
        train_dataset = XCO2WithAuxDataset(train_sequences, aux_dirs, input_size=args.input_size)
        val_dataset = XCO2WithAuxDataset(val_sequences, aux_dirs, input_size=args.input_size) if val_sequences else None
        # Determine input_channels by getting the shape of the first item
        # Determine input_channels by getting the shape of the first item
        if len(train_dataset) > 0:
            sample_input, _ = train_dataset[0]
            # 正确获取通道维度，输入形状应该是 [sequence_length, channels, height, width]
            print(f"Sample input shape: {sample_input.shape}")
            
            # 确保正确获取通道维度
            input_channels = sample_input.shape[1]  # 应该是第二个维度，而不是第三个
            print(f"Using {input_channels} input channels")
        else:
            input_channels = 1  # Default if dataset is empty
    else:
        # Use standard dataset with just XCO2
        print("Using only XCO2 data (no auxiliary features)")
        from step6_convlstm.data_loader import XCO2SequenceDataset
        train_dataset = XCO2SequenceDataset(train_sequences, input_size=args.input_size)
        val_dataset = XCO2SequenceDataset(val_sequences, input_size=args.input_size) if val_sequences else None
        input_channels = 1
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size) if val_dataset else None
    
    # Create model
    model = XCO2ConvLSTM(
        input_channels=input_channels,
        hidden_dims=args.hidden_dims,
        kernel_size=args.kernel_size,
        num_layers=len(args.hidden_dims),
        dropout=args.dropout
    )
    
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    # Define early stopping
    early_stopping = EarlyStopping(patience=args.patience)
    
    # Train model
    print("Starting training...")
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=args.epochs,
        scheduler=scheduler,
        early_stopping=early_stopping
    )
    
    # Plot training history
    plot_training_history(history, args.output_dir, args.experiment_name)
    
    # Load best model
    best_model_path = 'best_xco2_convlstm_model.pth'
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
        print(f"Loaded best model from {best_model_path}")
    
    # Visualize some predictions
    if val_loader is not None:
        visualize_prediction(model, val_loader, device, args.output_dir, args.experiment_name)
    
    # Save final model
    final_model_path = os.path.join(args.output_dir, f"final_{args.experiment_name}.pth")
    torch.save(model.state_dict(), final_model_path)
    print(f"Saved final model to {final_model_path}")
    
    # Save experiment configuration
    import json
    config_path = os.path.join(args.output_dir, f"config_{args.experiment_name}.json")
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=4)
    
    print(f"Saved experiment configuration to {config_path}")
    print("Training complete!")

if __name__ == "__main__":
    main() 
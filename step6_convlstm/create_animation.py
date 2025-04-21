"""
Create animation of XCO2 predictions from a series of TIF files.
This helps visualize the spatio-temporal patterns of XCO2 changes.
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import rasterio
from glob import glob
import re
from datetime import datetime
from matplotlib.colors import Normalize

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Create animation of XCO2 predictions')
    
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing XCO2 TIF files')
    parser.add_argument('--pattern', type=str, default='*_CONVLSTM_XCO2.tif',
                        help='Pattern to match TIF files, default is for ConvLSTM predictions')
    parser.add_argument('--output_path', type=str, default='xco2_animation.mp4',
                        help='Output path for the animation file')
    parser.add_argument('--fps', type=int, default=2,
                        help='Frames per second for the animation')
    parser.add_argument('--dpi', type=int, default=150,
                        help='DPI for the animation')
    parser.add_argument('--figsize', type=int, nargs=2, default=[10, 8],
                        help='Figure size in inches, default is 10x8')
    parser.add_argument('--cmap', type=str, default='viridis',
                        help='Colormap to use for visualization')
    parser.add_argument('--vmin', type=float, default=None,
                        help='Minimum value for colormap normalization')
    parser.add_argument('--vmax', type=float, default=None,
                        help='Maximum value for colormap normalization')
    parser.add_argument('--title_prefix', type=str, default='XCO2 Prediction:',
                        help='Prefix for frame titles')
    parser.add_argument('--sort_by_date', action='store_true',
                        help='Sort files by date extracted from filename')
    
    return parser.parse_args()

def extract_date_from_filename(filename):
    """
    Extract date from filename with format YYYY_MM_*.tif or similar.
    
    Args:
        filename (str): Filename to extract date from
        
    Returns:
        datetime.datetime: Extracted date, or None if not found
    """
    # Extract year and month using regex
    match = re.search(r'(\d{4})_(\d{1,2})', os.path.basename(filename))
    if match:
        year, month = map(int, match.groups())
        return datetime(year, month, 1)
    return None

def load_tif_files(data_dir, pattern='*.tif', sort_by_date=False):
    """
    Load all TIF files matching the pattern from the directory.
    
    Args:
        data_dir (str): Directory containing TIF files
        pattern (str): Glob pattern to match files
        sort_by_date (bool): Whether to sort files by date extracted from filename
        
    Returns:
        tuple: (data_list, metadata_list) arrays and metadata for each file
    """
    # Find all files matching the pattern
    file_paths = glob(os.path.join(data_dir, pattern))
    
    if not file_paths:
        raise ValueError(f"No files found matching pattern '{pattern}' in directory '{data_dir}'")
    
    # Sort files by date if requested
    if sort_by_date:
        dated_files = []
        for file_path in file_paths:
            date = extract_date_from_filename(file_path)
            if date:
                dated_files.append((date, file_path))
            else:
                print(f"Warning: Could not extract date from {os.path.basename(file_path)}")
        
        if dated_files:
            dated_files.sort(key=lambda x: x[0])
            file_paths = [f[1] for f in dated_files]
        else:
            print("Warning: Could not extract dates from any files, using alphabetical sorting")
            file_paths.sort()
    else:
        # Sort alphabetically
        file_paths.sort()
    
    # Load data from each file
    data_list = []
    metadata_list = []
    
    for file_path in file_paths:
        with rasterio.open(file_path) as src:
            # Read data and metadata
            data = src.read(1)
            metadata = {
                'filename': os.path.basename(file_path),
                'nodata': src.nodata,
                'date': extract_date_from_filename(file_path)
            }
            
            # Handle NoData values
            if src.nodata is not None:
                data = np.ma.masked_equal(data, src.nodata)
            
            data_list.append(data)
            metadata_list.append(metadata)
    
    return data_list, metadata_list

def create_animation(data_list, metadata_list, output_path, fps=2, dpi=150, 
                     figsize=(10, 8), cmap='viridis', vmin=None, vmax=None, 
                     title_prefix='XCO2 Prediction:'):
    """
    Create animation from a list of arrays.
    
    Args:
        data_list (list): List of 2D arrays to animate
        metadata_list (list): List of metadata dictionaries for each array
        output_path (str): Path to save the animation
        fps (int): Frames per second
        dpi (int): DPI for the animation
        figsize (tuple): Figure size in inches
        cmap (str): Colormap to use
        vmin (float): Minimum value for colormap normalization
        vmax (float): Maximum value for colormap normalization
        title_prefix (str): Prefix for frame titles
    """
    # Calculate global min/max if not provided
    if vmin is None or vmax is None:
        all_data = np.concatenate([data.flatten() for data in data_list if not np.ma.is_masked(data) or not np.all(data.mask)])
        global_vmin = np.nanmin(all_data) if vmin is None else vmin
        global_vmax = np.nanmax(all_data) if vmax is None else vmax
    else:
        global_vmin, global_vmax = vmin, vmax
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create a normalization instance
    norm = Normalize(vmin=global_vmin, vmax=global_vmax)
    
    # Function to update the frame
    def update_frame(i):
        ax.clear()
        im = ax.imshow(data_list[i], cmap=cmap, norm=norm)
        
        # Add title with date if available
        date_str = ''
        if metadata_list[i]['date']:
            date_str = metadata_list[i]['date'].strftime('%Y-%m')
        
        ax.set_title(f"{title_prefix} {date_str}")
        
        # Remove tick labels
        ax.set_xticks([])
        ax.set_yticks([])
        
        return [im]
    
    # Create animation
    ani = animation.FuncAnimation(
        fig, update_frame, frames=len(data_list), blit=True
    )
    
    # Add colorbar
    plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, label='XCO2 (ppm)')
    
    # Save animation
    writer = animation.FFMpegWriter(fps=fps)
    ani.save(output_path, writer=writer, dpi=dpi)
    plt.close(fig)
    
    print(f"Animation saved to {output_path}")

def main():
    """Main function."""
    # Parse command line arguments
    args = parse_args()
    
    # Load TIF files
    try:
        data_list, metadata_list = load_tif_files(
            args.data_dir, args.pattern, args.sort_by_date
        )
        print(f"Loaded {len(data_list)} files for animation")
    except ValueError as e:
        print(f"Error: {e}")
        return
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Create animation
    create_animation(
        data_list, metadata_list, args.output_path,
        fps=args.fps, dpi=args.dpi, figsize=tuple(args.figsize),
        cmap=args.cmap, vmin=args.vmin, vmax=args.vmax,
        title_prefix=args.title_prefix
    )

if __name__ == "__main__":
    main() 
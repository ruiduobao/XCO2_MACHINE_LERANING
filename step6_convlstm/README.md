训练代码：
python train.py --data_dir "E:\地理所\论文\中国XCO2论文_2025.04\数据\XGBOOST_XCO2" --sequence_length 3 --hidden_dims 32 64 --aux_data --batch_size 2 --epochs 50 --output_dir ./results --experiment_name xco2_convlstm_6month
记得删掉有一些不重要的特征，加快收敛速度

预测代码：
python predict_2018.py --sequence_length 3 --start_date 2018-03 --output_dir "./2018预测结果" --num_steps 9 --visualize


考虑了空值的训练代码：
python train.py --data_dir "E:\地理所\论文\中国XCO2论文_2025.04\数据\XGBOOST_XCO2" --sequence_length 3 --batch_size 2 --epochs 2 --output_dir "./results" --experiment_name "convlstm_china_only" --aux_data 
考虑了空值的预测代码：
python predict_aux.py --model_path "E:\地理所\论文\中国XCO2论文_2025.04\代码\step6_convlstm\已训练模型\best_xco2_convlstm_model.pth" --data_dir "E:\地理所\论文\中国XCO2论文_2025.04\数据\XGBOOST_XCO2" --sequence_length 3 --start_date "2018-01" --num_steps 12 --output_dir "./predictions_2018_with_aux" --visualize



# ConvLSTM for XCO2 Spatio-Temporal Prediction

This module implements a Convolutional LSTM (ConvLSTM) model for spatio-temporal prediction of XCO2 concentrations. The model can capture both spatial and temporal correlations in the data to make accurate future predictions.

## Overview

ConvLSTM is a recurrent neural network that incorporates convolutional operations within the LSTM cell. It is specifically designed for spatio-temporal sequence prediction tasks and is well-suited for predicting future XCO2 distribution based on past patterns.

Key features:
- Captures spatial correlations using convolutional operations
- Models temporal dynamics with LSTM memory cells
- Supports multi-channel inputs for incorporating auxiliary data
- Produces high-quality predictions of future XCO2 distributions

## Installation Requirements

The implementation requires the following Python packages:

```
torch>=1.7.0
rasterio>=1.2.0
numpy>=1.19.0
matplotlib>=3.3.0
tqdm>=4.50.0
scikit-image>=0.17.0
pandas>=1.1.0
python-dateutil>=2.8.0
```

You can install them using pip:

```bash
pip install torch rasterio numpy matplotlib tqdm scikit-image pandas python-dateutil
```

## Module Structure

- `data_loader.py`: Utilities for loading and preprocessing XCO2 and auxiliary data
- `model.py`: Implementation of the ConvLSTM model architecture
- `train.py`: Script for training the ConvLSTM model
- `predict.py`: Script for making predictions with a trained model

## Data Format

The model expects XCO2 data as GeoTIFF files. The filename format should be:
`YYYY_MM_XGBOOST_XCO2.tif` (e.g., `2018_03_XGBOOST_XCO2.tif`)

Auxiliary data can also be included to improve predictions. Each auxiliary file should follow a consistent naming convention, typically:

For monthly data: `FEATURE_YYYY_MM.tif` (e.g., `VIIRS_2018_03.tif`)
For annual data: `FEATURE_YYYY.tif` (e.g., `DEM_2018.tif`)

## Training the Model

To train the ConvLSTM model on XCO2 data:

```bash
python train.py --data_dir /path/to/xgboost_xco2_tifs --sequence_length 6 --hidden_dims 32 64 --output_dir ./results
```

Key parameters:
- `--data_dir`: Directory containing XCO2 TIF files
- `--sequence_length`: Number of time steps (months) to use as input
- `--hidden_dims`: Hidden dimensions for each ConvLSTM layer
- `--aux_data`: Flag to include auxiliary data features
- `--batch_size`: Batch size for training
- `--lr`: Learning rate
- `--epochs`: Number of training epochs
- `--gpu`: GPU index to use (-1 for CPU)

Example with auxiliary data and custom parameters:

```bash
python train.py --data_dir /path/to/xgboost_xco2_tifs --sequence_length 12 --hidden_dims 64 128 --aux_data --batch_size 8 --lr 0.0005 --epochs 100
```

## Making Predictions

Once you have a trained model, you can use it to make predictions:

```bash
python predict.py --model_path ./results/final_model.pth --data_dir /path/to/xgboost_xco2_tifs --sequence_length 6 --num_steps 12 --output_dir ./predictions --visualize
```

Key parameters:
- `--model_path`: Path to the trained model weights
- `--config_path`: (Optional) Path to model configuration file
- `--start_date`: Start date for prediction in YYYY-MM format
- `--num_steps`: Number of steps (months) to predict ahead
- `--visualize`: Flag to generate visualizations of predictions

## Examples

### Training Example

Train a model with 6-month input sequences and 2 ConvLSTM layers:

```bash
python train.py --data_dir "E:\地理所\论文\中国XCO2论文_2025.04\数据\XGBOOST_XCO2" --sequence_length 6 --hidden_dims 32 64 --aux_data --batch_size 4 --epochs 50 --output_dir ./results --experiment_name xco2_convlstm_6month
```

### Prediction Example

Make predictions for the next 12 months using the trained model:

```bash
python predict.py --model_path ./results/final_xco2_convlstm_6month.pth --data_dir "E:\地理所\论文\中国XCO2论文_2025.04\数据\XGBOOST_XCO2" --sequence_length 6 --start_date 2021-12 --num_steps 12 --output_dir ./predictions --visualize
```

## Model Architecture

The ConvLSTM model consists of:

1. Input layer: Takes a sequence of XCO2 maps (optionally with auxiliary data)
2. ConvLSTM layers: Processes the input sequence and captures spatio-temporal patterns
3. Convolutional decoder: Transforms the final hidden state to produce a prediction

The implementation supports multiple stacked ConvLSTM layers with different hidden dimensions, and includes dropout and batch normalization for better regularization.

## Citation

If you use this implementation in your research, please cite:

```
Shi, X., Chen, Z., Wang, H., Yeung, D., Wong, W., & Woo, W. (2015). 
Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting. 
In Advances in Neural Information Processing Systems (pp. 802-810).
```

## License

This project is open source and available under the [MIT License](LICENSE). 



# Understanding the ConvLSTM Model for XCO2 Prediction

The codebase implements a Convolutional LSTM (ConvLSTM) model for predicting XCO2 (carbon dioxide concentration) in a spatio-temporal context. Here's a detailed breakdown:

## 1. Model Architecture

The model is defined in `model.py` and consists of:

### Core Components:
- **ConvLSTMCell**: A basic cell combining convolutional operations with LSTM gates
- **ConvLSTM**: A multi-layer ConvLSTM network that processes sequences of spatial data
- **XCO2ConvLSTM**: The complete model architecture that combines:
  - ConvLSTM encoder for capturing spatio-temporal dynamics
  - Dropout and batch normalization for regularization
  - Convolutional decoder for transforming hidden states to predicted XCO2 maps

### Key Architecture Details:
- The model takes input tensor of shape `(batch_size, seq_len, channels, height, width)`
- Processes through multiple ConvLSTM layers with specified hidden dimensions
- Uses the hidden state from the last time step as input to the decoder
- The decoder reduces the number of channels through convolutional layers to output a single-channel XCO2 prediction

## 2. Data Processing Flow

### Input Data:
- **XCO2 Data**: Primary time-series of XCO2 maps
- **Auxiliary Data** (optional): Additional features like:
  - Geographic: Latitude, Longitude, DEM, slope, aspect
  - Environmental: ERA5Land, AOD, NDVI
  - Human activity: VIIRS, landscan, human footprint, CLCD, MODIS landcover
  - Other CO2 sources: CAMS, carbon tracer, ODIAC

### Data Loading and Processing:
- `data_loader.py` contains the data processing pipeline:
  - **XCO2SequenceDataset**: Handles basic XCO2 sequence data
  - **XCO2WithAuxDataset**: Enhances sequences with auxiliary features
  - Functions for listing, loading, and normalizing TIF files
  - Creating sequences of appropriate length from the time series

### Sequence Creation:
- The model uses a sliding window approach to create training sequences
- Each sequence consists of multiple consecutive time steps (e.g., 6 months)
- The target is the XCO2 map for the time step immediately following the sequence

## 3. Training Process

The training process in `train.py` follows these steps:

1. **Parse Arguments**: Set model parameters, data paths, and training settings
2. **Load Data**: Create training and validation sequences and dataloaders
3. **Initialize Model**: Create a ConvLSTM model with specified parameters
4. **Train Model**:
   - Loop through epochs and batches
   - Calculate loss using MSE between predictions and targets
   - Backpropagate and update model parameters
   - Validate on validation set
   - Apply learning rate scheduling and early stopping
5. **Save Results**: Save the best model, training history, and visualizations

## 4. Prediction Process

The prediction workflow in `predict.py` includes:

1. **Load Model**: Load a trained model with weights from a specified path
2. **Prepare Input Sequence**: Create a sequence from historical data
3. **Generate Predictions**: Two prediction modes:
   - **Single-step prediction**: Predict the next time step
   - **Multi-step prediction**: Iteratively predict multiple future time steps
4. **Post-processing**: Denormalize predictions to original value range
5. **Save Results**: Save predictions as GeoTIFF files and optionally visualize them

## 5. Key Features and Innovations

- **Spatio-temporal Modeling**: Captures both spatial patterns and temporal dynamics
- **Multi-scale Feature Learning**: Through ConvLSTM with different hidden dimensions
- **Auxiliary Data Integration**: Can incorporate multiple data sources beyond XCO2
- **Flexible Configuration**: Customizable parameters for model architecture and training
- **Visualization Tools**: Functions to visualize predictions and training history

## 6. Usage Example

To train a model:
```
python -m step6_convlstm.train --data_dir [XCO2_DIR] --aux_data --sequence_length 6 --hidden_dims 32 64 --epochs 50
```

To make predictions:
```
python -m step6_convlstm.predict --model_path [MODEL_PATH] --data_dir [XCO2_DIR] --aux_data --start_date 2020-12 --num_steps 12
```

The ConvLSTM model offers a sophisticated approach to XCO2 prediction by capturing the complex spatio-temporal patterns in carbon dioxide concentration. Its ability to incorporate auxiliary data makes it particularly powerful for understanding the relationship between XCO2 and various environmental and human factors.

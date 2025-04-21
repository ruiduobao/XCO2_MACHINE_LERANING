"""
ConvLSTM for XCO2 Spatio-Temporal Prediction

This module implements a Convolutional LSTM (ConvLSTM) model for predicting
XCO2 concentrations over time and space based on historical data.
"""

# Import core functionality for direct use
from step6_convlstm.model import (
    ConvLSTMCell, 
    ConvLSTM, 
    XCO2ConvLSTM, 
    train_model, 
    predict_next_step, 
    predict_multiple_steps,
    EarlyStopping
)

from step6_convlstm.data_loader import (
    list_xco2_files,
    load_tif_file,
    create_sequence_data,
    XCO2SequenceDataset,
    XCO2WithAuxDataset,
    get_dataloaders,
    create_auxiliary_input
)

# Main package info
__version__ = '1.0.0'
__author__ = 'China XCO2 Research Team' 
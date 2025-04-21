import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ConvLSTMCell(nn.Module):
    """
    Basic ConvLSTM cell as described in the paper:
    "Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting"
    by Xingjian Shi et al.
    """
    
    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True):
        """
        Initialize ConvLSTM cell.
        
        Args:
            input_dim (int): Number of channels of input tensor.
            hidden_dim (int): Number of channels of hidden state.
            kernel_size (int or tuple): Size of the convolutional kernel.
            bias (bool): Whether to add bias to the convolution layers.
        """
        super(ConvLSTMCell, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Ensure kernel_size is a tuple
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size
            
        self.padding = (self.kernel_size[0] // 2, self.kernel_size[1] // 2)
        self.bias = bias
        
        # Four convolutional layers for various gates
        # All gates are computed with a single convolution, but split into four parts
        self.conv = nn.Conv2d(
            in_channels=self.input_dim + self.hidden_dim,
            out_channels=4 * self.hidden_dim,  # For input, forget, cell, output gates
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias
        )
    
    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        
        # Concatenate input and previous hidden state
        combined = torch.cat([input_tensor, h_cur], dim=1)
        
        # Calculate all gates in one convolution
        combined_conv = self.conv(combined)
        
        # Split the combined convolution output into the four gates
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        
        # Apply activations to get gate values
        i = torch.sigmoid(cc_i)  # Input gate
        f = torch.sigmoid(cc_f)  # Forget gate
        o = torch.sigmoid(cc_o)  # Output gate
        g = torch.tanh(cc_g)     # Cell input
        
        # Update cell state and hidden state
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        
        return h_next, c_next
    
    def init_hidden(self, batch_size, height, width):
        """
        Initialize hidden state and cell state with zeros.
        
        Args:
            batch_size (int): Batch size
            height (int): Height of feature map
            width (int): Width of feature map
            
        Returns:
            tuple: Initial hidden state and cell state
        """
        device = self.conv.weight.device
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=device))


class ConvLSTM(nn.Module):
    """
    ConvLSTM layer with multiple cells stacked in the time dimension.
    """
    
    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers=1, 
                 batch_first=True, bias=True, return_all_layers=False):
        """
        Initialize a ConvLSTM module.
        
        Args:
            input_dim (int): Number of channels of input tensor.
            hidden_dim (int or list): Number of channels of hidden state.
                If it's a list, num_layers must match its length.
            kernel_size (int or tuple): Size of the convolutional kernel.
            num_layers (int): Number of ConvLSTM layers stacked.
            batch_first (bool): If True, the input and output tensors are provided
                as (batch, seq, channel, height, width) instead of (seq, batch, channel, height, width).
            bias (bool): Whether to add bias to the convolution layers.
            return_all_layers (bool): If True, returns outputs of all layers,
                otherwise just the last layer.
        """
        super(ConvLSTM, self).__init__()
        
        self._check_kernel_size_consistency(kernel_size)
        
        # Make sure hidden_dim is a list with len = num_layers
        if not isinstance(hidden_dim, list):
            self.hidden_dim = [hidden_dim] * num_layers
        else:
            if len(hidden_dim) != num_layers:
                raise ValueError(f"Length of hidden_dim list ({len(hidden_dim)}) must match num_layers ({num_layers})")
            self.hidden_dim = hidden_dim
        
        self.input_dim = input_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers
        
        # Create ConvLSTM cells
        cell_list = []
        for i in range(self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i-1]
            cell_list.append(
                ConvLSTMCell(
                    input_dim=cur_input_dim,
                    hidden_dim=self.hidden_dim[i],
                    kernel_size=self.kernel_size,
                    bias=self.bias
                )
            )
            
        self.cell_list = nn.ModuleList(cell_list)
    
    def forward(self, input_tensor, hidden_state=None):
        """
        Forward pass through the ConvLSTM network.
        
        Args:
            input_tensor: Input of shape (b, t, c, h, w) or (t, b, c, h, w) depending on batch_first.
            hidden_state: Initial hidden state. If None, initialized as zeros.
            
        Returns:
            layer_output_list: List of outputs from each layer
            last_state_list: List of final hidden states from each layer
        """
        # Ensure input_tensor is 5D
        if input_tensor.dim() != 5:
            raise ValueError(f"Expected 5D input (got {input_tensor.dim()}D input)")
        
        # If batch_first, transpose to seq_first for internal calculations
        if self.batch_first:
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)
            # Now input_tensor is (t, b, c, h, w)
            
        # Get dimensions
        b, _, _, h, w = input_tensor.size()
        seq_len = input_tensor.size(0)
        
        # Initialize hidden states if not provided
        if hidden_state is None:
            hidden_state = self._init_hidden(batch_size=b, height=h, width=w)
            
        layer_output_list = []
        last_state_list = []
        
        # Process each layer
        cur_layer_input = input_tensor
        
        for layer_idx in range(self.num_layers):
            h, c = hidden_state[layer_idx]
            output_inner = []
            
            # Process each time step
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](
                    input_tensor=cur_layer_input[t],
                    cur_state=[h, c]
                )
                output_inner.append(h)
                
            # Stack outputs along sequence dimension
            layer_output = torch.stack(output_inner, dim=0)
            cur_layer_input = layer_output
            
            layer_output_list.append(layer_output)
            last_state_list.append([h, c])
            
        # If batch_first, transpose back
        if self.batch_first:
            layer_output_list = [output.permute(1, 0, 2, 3, 4) for output in layer_output_list]
            
        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]
            
        return layer_output_list, last_state_list
    
    def _init_hidden(self, batch_size, height, width):
        """
        Initialize hidden states for all layers.
        
        Args:
            batch_size (int): Batch size
            height (int): Height of feature map
            width (int): Width of feature map
            
        Returns:
            list: List of initial hidden and cell states for each layer
        """
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, height, width))
        return init_states
    
    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        """
        Check kernel size consistency (make sure it's a tuple).
        
        Args:
            kernel_size: int or tuple
        """
        if not (isinstance(kernel_size, tuple) or isinstance(kernel_size, list) or
                isinstance(kernel_size, int)):
            raise ValueError("kernel_size must be an int or a tuple/list of ints")
        
        if isinstance(kernel_size, int):
            return (kernel_size, kernel_size)
        else:
            assert len(kernel_size) == 2, "kernel_size must be a tuple/list of length 2"
            return kernel_size


class XCO2ConvLSTM(nn.Module):
    """
    ConvLSTM model for XCO2 prediction.
    
    Architecture:
    1. Input layer: Takes a sequence of XCO2 maps (optionally with auxiliary data)
    2. ConvLSTM encoder: Captures spatio-temporal dynamics
    3. Convolutional decoder: Transforms hidden state to predicted XCO2 map
    """
    
    def __init__(self, input_channels=1, hidden_dims=[32, 64], kernel_size=3, 
                 num_layers=2, dropout=0.2, batch_norm=True):
        """
        Initialize XCO2ConvLSTM model.
        
        Args:
            input_channels (int): Number of input channels (1 for XCO2 only, more if auxiliary data is used)
            hidden_dims (list): List of hidden dimensions for each ConvLSTM layer
            kernel_size (int or tuple): Size of the convolutional kernel in ConvLSTM
            num_layers (int): Number of ConvLSTM layers
            dropout (float): Dropout probability
            batch_norm (bool): Whether to use batch normalization
        """
        super(XCO2ConvLSTM, self).__init__()
        
        # Ensure hidden_dims has the right length
        if len(hidden_dims) != num_layers:
            raise ValueError(f"Length of hidden_dims ({len(hidden_dims)}) must match num_layers ({num_layers})")
            
        self.input_channels = input_channels
        self.hidden_dims = hidden_dims
        self.batch_norm = batch_norm
        
        # ConvLSTM encoder
        self.encoder = ConvLSTM(
            input_dim=input_channels,
            hidden_dim=hidden_dims,
            kernel_size=kernel_size,
            num_layers=num_layers,
            batch_first=True,
            bias=True,
            return_all_layers=False
        )
        
        # Dropout
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else None
        
        # Batch normalization
        self.bn = nn.BatchNorm2d(hidden_dims[-1]) if batch_norm else None
        
        # Convolutional decoder
        # Gradually reduce channels from hidden_dim to 1 (XCO2)
        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_dims[-1], hidden_dims[-1] // 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(hidden_dims[-1] // 2) if batch_norm else nn.Identity(),
            nn.Conv2d(hidden_dims[-1] // 2, 1, kernel_size=1),
        )
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, channels, height, width)
            
        Returns:
            torch.Tensor: Predicted XCO2 map for the next time step
        """
        # Pass through ConvLSTM encoder
        layer_outputs, last_states = self.encoder(x)
        
        # Get the output from the last layer
        # last_states is a list of [h, c] pairs for each layer
        h_last = last_states[0][0]  # Get the hidden state from the last layer
        
        # Apply dropout if specified
        if self.dropout is not None:
            h_last = self.dropout(h_last)
            
        # Apply batch normalization if specified
        if self.bn is not None:
            h_last = self.bn(h_last)
        
        # Pass through convolutional decoder
        output = self.decoder(h_last)
        
        return output


# Training utilities
def train_model(model, train_loader, val_loader, criterion, optimizer, 
                device, num_epochs=10, scheduler=None, early_stopping=None):
    """
    Train the XCO2ConvLSTM model.
    
    Args:
        model (nn.Module): Model to train
        train_loader (DataLoader): Training data loader
        val_loader (DataLoader): Validation data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on (cuda or cpu)
        num_epochs (int): Number of epochs to train for
        scheduler: Learning rate scheduler (optional)
        early_stopping: Early stopping callback (optional)
        
    Returns:
        dict: Training history
    """
    # Move model to device
    model = model.to(device)
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'best_epoch': 0,
        'best_val_loss': float('inf')
    }
    
    # Train for the specified number of epochs
    for epoch in range(num_epochs):
        # Set model to training mode
        model.train()
        train_loss = 0.0
        
        # Training loop
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            # Move data to device
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            
            # Calculate loss
            loss = criterion(outputs, targets)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Accumulate loss
            train_loss += loss.item()
            
            # Print progress
            if (batch_idx + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(train_loader)}], '
                      f'Loss: {loss.item():.4f}')
        
        # Calculate average training loss for the epoch
        train_loss /= len(train_loader)
        history['train_loss'].append(train_loss)
        
        # Validation
        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        
        # Calculate average validation loss
        val_loss /= len(val_loader)
        history['val_loss'].append(val_loss)
        
        # Print epoch summary
        print(f'Epoch [{epoch+1}/{num_epochs}], '
              f'Train Loss: {train_loss:.4f}, '
              f'Val Loss: {val_loss:.4f}')
        
        # Update learning rate if scheduler is provided
        if scheduler is not None:
            scheduler.step(val_loss)
        
        # Check if this is the best model so far
        if val_loss < history['best_val_loss']:
            history['best_val_loss'] = val_loss
            history['best_epoch'] = epoch
            # Save the best model
            torch.save(model.state_dict(), 'best_xco2_convlstm_model.pth')
            print(f"New best model saved with validation loss: {val_loss:.4f}")
        
        # Early stopping if specified
        if early_stopping is not None and early_stopping(val_loss):
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    return history


# Prediction utilities
def predict_next_step(model, sequence, device='cuda'):
    """
    Predict the next time step given a sequence.
    
    Args:
        model (nn.Module): Trained model
        sequence (torch.Tensor): Input sequence tensor of shape (1, seq_len, channels, height, width)
        device: Device to run prediction on
        
    Returns:
        numpy.ndarray: Predicted map for the next time step
    """
    model.eval()
    with torch.no_grad():
        sequence = sequence.to(device)
        prediction = model(sequence)
        # Squeeze to remove batch and channel dimensions for a single prediction
        prediction = prediction.squeeze().cpu().numpy()
    return prediction


def predict_multiple_steps(model, initial_sequence, num_steps, device='cuda'):
    """
    Predict multiple time steps ahead by feeding predictions back as inputs.
    
    Args:
        model (nn.Module): Trained model
        initial_sequence (torch.Tensor): Initial input sequence tensor of shape (1, seq_len, channels, height, width)
        num_steps (int): Number of steps to predict ahead
        device: Device to run prediction on
        
    Returns:
        list: List of predicted maps for each future time step
    """
    model.eval()
    
    # Get sequence parameters
    sequence_length = initial_sequence.shape[1]
    
    # Copy the initial sequence to avoid modifying it
    sequence = initial_sequence.clone().to(device)
    
    # List to store predictions
    predictions = []
    
    with torch.no_grad():
        for _ in range(num_steps):
            # Predict next step
            next_step = model(sequence)
            
            # Store prediction
            predictions.append(next_step.squeeze().cpu().numpy())
            
            # Update sequence by removing the first time step and adding the prediction
            sequence = sequence.clone()
            sequence = torch.cat([sequence[:, 1:], next_step.unsqueeze(1)], dim=1)
    
    return predictions


# Early stopping utility
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        """
        Initialize early stopping.
        
        Args:
            patience (int): Number of epochs to wait after validation loss stops improving
            min_delta (float): Minimum change in validation loss to be considered an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss):
        """
        Check if early stopping criteria is met.
        
        Args:
            val_loss (float): Current validation loss
            
        Returns:
            bool: True if early stopping criteria is met, False otherwise
        """
        if self.best_loss is None:
            self.best_loss = val_loss
            return False
        
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        
        return False 
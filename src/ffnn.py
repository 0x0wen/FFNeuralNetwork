import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional, Union, Callable
import logging
import time
import os
from datetime import datetime


class FFNN:
    """
    Feedforward Neural Network (FFNN) with comprehensive logging
    """
    
    def __init__(
        self, 
        layer_sizes: List[int],
        activation_functions: List[str],
        loss_function: str,
        weight_initialization: str,
        weight_init_params: Dict,
        logging_level: str = "INFO",
        log_file: Optional[str] = None
    ):
        if len(layer_sizes) < 2:
            raise ValueError("At least 2 layers (input and output) are required")
        
        if len(activation_functions) != len(layer_sizes) - 1:
            raise ValueError("Number of activation functions must be equal to number of layers - 1")
        
        # Set up logging
        self._setup_logging(logging_level, log_file)
        
        self.layer_sizes = layer_sizes
        self.n_layers = len(layer_sizes)
        self.activation_functions = activation_functions
        self.loss_function = loss_function
        
        self.weights = []
        self.biases = []
        self.gradients_w = []
        self.gradients_b = []
        
        self.logger.info(f"Initializing FFNN with {self.n_layers} layers: {layer_sizes}")
        self.logger.info(f"Activation functions: {activation_functions}")
        self.logger.info(f"Loss function: {loss_function}")
        self.logger.info(f"Weight initialization method: {weight_initialization}")
        
        self._initialize_weights(weight_initialization, weight_init_params)
        
        self.z_values = []  
        self.a_values = []  
        
        self.history = {
            'train_loss': [],
            'val_loss': []
        }
    
    def _setup_logging(self, logging_level: str, log_file: Optional[str] = None):
        """Set up logger for the FFNN instance"""
        # Create logger
        self.logger = logging.getLogger(f"FFNN_{id(self)}")
        self.logger.setLevel(getattr(logging, logging_level.upper()))
        
        # Remove any existing handlers to avoid duplicates
        if self.logger.hasHandlers():
            self.logger.handlers.clear()
            
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # Create file handler if log_file is specified
        if log_file:
            # Create log directory if it doesn't exist
            log_dir = os.path.dirname(log_file)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir)
                
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
            
        self.logger.info("Logger initialized")
    
    def _initialize_weights(self, method: str, params: Dict):
        self.logger.info(f"Initializing weights using {method} method with params {params}")
        np.random.seed(params.get('seed', 42))
        
        for i in range(self.n_layers - 1):
            input_size = self.layer_sizes[i]
            output_size = self.layer_sizes[i + 1]
            
            self.logger.debug(f"Layer {i+1}: Initializing weights of shape ({input_size}, {output_size})")
            
            if method.lower() == "zeros":
                W = np.zeros((input_size, output_size))
                b = np.zeros((1, output_size))
            
            elif method.lower() == "random_uniform":
                lower_bound = params.get('lower_bound', -0.1)  
                upper_bound = params.get('upper_bound', 0.1)   
                W = np.random.uniform(low=lower_bound, high=upper_bound, size=(input_size, output_size))
                b = np.random.uniform(low=lower_bound, high=upper_bound, size=(1, output_size))
            
            elif method.lower() == "random_normal":
                mean = params.get('mean', 0.0)         
                variance = params.get('variance', 0.1) 
                std = np.sqrt(variance)                
                W = np.random.normal(loc=mean, scale=std, size=(input_size, output_size))
                b = np.random.normal(loc=mean, scale=std, size=(1, output_size))
            
            elif method.lower() == "xavier":
                limit = np.sqrt(6 / (input_size + output_size))
                W = np.random.uniform(low=-limit, high=limit, size=(input_size, output_size))
                b = np.zeros((1, output_size))  
                
            elif method.lower() == "he":
                limit = np.sqrt(2 / input_size)
                W = np.random.normal(loc=0.0, scale=limit, size=(input_size, output_size))
                b = np.zeros((1, output_size))  
            
            else:
                raise ValueError(f"Method Intialize '{method}' not supported. Chose: 'zeros', 'random_uniform', 'random_normal', 'xavier', 'he'.")
            
            self.weights.append(W)
            self.biases.append(b)
            
            self.gradients_w.append(np.zeros_like(W))
            self.gradients_b.append(np.zeros_like(b))
            
            self.logger.debug(f"Layer {i+1}: Weight stats - Mean: {np.mean(W):.6f}, Std: {np.std(W):.6f}, Min: {np.min(W):.6f}, Max: {np.max(W):.6f}")
            self.logger.debug(f"Layer {i+1}: Bias stats - Mean: {np.mean(b):.6f}, Std: {np.std(b):.6f}, Min: {np.min(b):.6f}, Max: {np.max(b):.6f}")

    
    def _activation_forward(self, Z: np.ndarray, activation: str) -> np.ndarray:
        self.logger.debug(f"Applying {activation} activation function")
        
        if activation == "linear":
            return Z
        
        elif activation == "sigmoid":
            return 1 / (1 + np.exp(-np.clip(Z, -500, 500))) 
        
        elif activation == "tanh":
            return np.tanh(Z)
        
        elif activation == "relu":
            result = np.maximum(0, Z)
            zeros_percentage = np.mean(result == 0) * 100
            self.logger.debug(f"ReLU zeros: {zeros_percentage:.2f}% of neurons inactive")
            return result
        
        elif activation == "leaky_relu":
            return np.maximum(0.01 * Z, Z)
        
        elif activation == "elu":
            return np.where(Z > 0, Z, np.exp(Z) - 1)
        
        elif activation == "softmax":
            shifted_Z = Z - np.max(Z, axis=1, keepdims=True)
            exp_Z = np.exp(shifted_Z)
            return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)
        
        else:
            raise ValueError(f"Unsupported activation function: {activation}")
    
    def _activation_backward(self, dA: np.ndarray, Z: np.ndarray, activation: str) -> np.ndarray:
        self.logger.debug(f"Computing gradient for {activation} activation")
        
        if activation == "linear":
            return dA
        
        elif activation == "sigmoid":
            A = 1 / (1 + np.exp(-np.clip(Z, -500, 500)))
            return dA * A * (1 - A)
        
        elif activation == "tanh":
            return dA * (1 - np.tanh(Z) ** 2)
        
        elif activation == "relu":
            dZ = dA.copy()
            dZ[Z <= 0] = 0
            zeros_percentage = np.mean(dZ == 0) * 100
            self.logger.debug(f"ReLU backward zeros: {zeros_percentage:.2f}% of gradients are zero")
            return dZ
        
        elif activation == "leaky_relu":
            dZ = dA.copy()
            dZ[Z <= 0] *= 0.01
            return dZ
        
        elif activation == "elu":
            dZ = dA.copy()
            dZ[Z <= 0] *= np.exp(Z[Z <= 0])
            return dZ
        
        elif activation == "softmax":
            # For softmax, we typically compute dZ directly in backward pass
            # when used with categorical crossentropy
            return dA
        
        else:
            raise ValueError(f"Unsupported activation function: {activation}")
    
    def _compute_loss(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        self.logger.debug(f"Computing {self.loss_function} loss")
        
        if self.loss_function == "mse":
            loss = np.mean(np.square(y_pred - y_true))
            self.logger.debug(f"MSE loss: {loss:.6f}")
            return loss
        
        elif self.loss_function == "binary_crossentropy":
            y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
            loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
            self.logger.debug(f"Binary cross-entropy loss: {loss:.6f}")
            return loss
        
        elif self.loss_function == "categorical_crossentropy":
            y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
            loss = -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
            self.logger.debug(f"Categorical cross-entropy loss: {loss:.6f}")
            return loss
        
        else:
            raise ValueError(f"Unsupported loss function: {self.loss_function}")
    
    def _compute_loss_gradient(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        m = y_true.shape[0]  
        self.logger.debug(f"Computing gradient of {self.loss_function} loss")
        
        if self.loss_function == "mse":
            return (2/m) * (y_pred - y_true)
        
        elif self.loss_function == "binary_crossentropy":
            y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
            return (1/m) * (- y_true / y_pred + (1 - y_true) / (1 - y_pred))
        
        elif self.loss_function == "categorical_crossentropy":
            if self.activation_functions[-1] == "softmax":
                gradient = (1/m) * (y_pred - y_true)
                self.logger.debug(f"Loss gradient stats - Mean: {np.mean(gradient):.6f}, Std: {np.std(gradient):.6f}")
                return gradient
            else:
                y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
                return (1/m) * (- y_true / y_pred)
        
        else:
            raise ValueError(f"Unsupported loss function: {self.loss_function}")
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        self.logger.debug(f"Forward pass started with input shape {X.shape}")
        start_time = time.time()
        
        self.z_values = []
        self.a_values = [X] 
        
        A = X
        for i in range(self.n_layers - 1):
            W = self.weights[i]
            b = self.biases[i]
            activation = self.activation_functions[i]
            
            self.logger.debug(f"Layer {i+1}: Forward computation with {activation} activation")
            self.logger.debug(f"Layer {i+1}: Input shape: {A.shape}, Weight shape: {W.shape}, Bias shape: {b.shape}")
            
            Z = np.dot(A, W) + b
            
            # Log statistics for pre-activation values
            self.logger.debug(f"Layer {i+1}: Z stats - Mean: {np.mean(Z):.6f}, Std: {np.std(Z):.6f}, Min: {np.min(Z):.6f}, Max: {np.max(Z):.6f}")
            
            A = self._activation_forward(Z, activation)
            
            # Log statistics for post-activation values
            self.logger.debug(f"Layer {i+1}: A stats - Mean: {np.mean(A):.6f}, Std: {np.std(A):.6f}, Min: {np.min(A):.6f}, Max: {np.max(A):.6f}")
            
            self.z_values.append(Z)
            self.a_values.append(A)
        
        elapsed = time.time() - start_time
        self.logger.debug(f"Forward pass completed in {elapsed:.6f} seconds, output shape: {A.shape}")
        
        return A  
    
    def backward(self, y_true: np.ndarray) -> None:
        self.logger.debug("Backward pass started")
        start_time = time.time()
        
        m = y_true.shape[0]
        y_pred = self.a_values[-1]
        
        # Calculate initial gradient from loss function
        dA = self._compute_loss_gradient(y_pred, y_true)
        
        # Backpropagation for each layer
        for layer in reversed(range(self.n_layers - 1)):
            self.logger.debug(f"Layer {layer+1}: Backward computation")
            
            Z = self.z_values[layer]
            A_prev = self.a_values[layer]
            W = self.weights[layer]
            activation = self.activation_functions[layer]
            
            # Calculate gradient of activation function
            dZ = self._activation_backward(dA, Z, activation)
            
            # Calculate gradients for weights and biases
            dW = (1/m) * np.dot(A_prev.T, dZ)
            db = (1/m) * np.sum(dZ, axis=0, keepdims=True)
            
            # Store gradients
            self.gradients_w[layer] = dW
            self.gradients_b[layer] = db
            
            # Log gradient statistics
            self.logger.debug(f"Layer {layer+1}: dW stats - Mean: {np.mean(dW):.6f}, Std: {np.std(dW):.6f}, Min: {np.min(dW):.6f}, Max: {np.max(dW):.6f}")
            self.logger.debug(f"Layer {layer+1}: db stats - Mean: {np.mean(db):.6f}, Std: {np.std(db):.6f}, Min: {np.min(db):.6f}, Max: {np.max(db):.6f}")
            
            # Calculate gradient for previous layer
            if layer > 0:
                dA = np.dot(dZ, W.T)
                self.logger.debug(f"Layer {layer}: dA stats - Mean: {np.mean(dA):.6f}, Std: {np.std(dA):.6f}")
        
        elapsed = time.time() - start_time
        self.logger.debug(f"Backward pass completed in {elapsed:.6f} seconds")

    def update_weights(self, learning_rate: float) -> None:
        self.logger.debug(f"Updating weights with learning rate: {learning_rate}")
        start_time = time.time()
        
        # Update weights and biases based on calculated gradients
        for i in range(self.n_layers - 1):
            # Get gradients
            dW = self.gradients_w[i]
            db = self.gradients_b[i]
            
            # Get current parameters
            W = self.weights[i]
            b = self.biases[i]
            
            # Calculate weight update
            W_update = learning_rate * dW
            b_update = learning_rate * db
            
            # Log update statistics
            self.logger.debug(f"Layer {i+1}: Weight update stats - Mean: {np.mean(W_update):.6f}, Max: {np.max(np.abs(W_update)):.6f}")
            self.logger.debug(f"Layer {i+1}: Bias update stats - Mean: {np.mean(b_update):.6f}, Max: {np.max(np.abs(b_update)):.6f}")
            
            # Update parameters
            self.weights[i] = W - W_update
            self.biases[i] = b - b_update
        
        elapsed = time.time() - start_time
        self.logger.debug(f"Weight update completed in {elapsed:.6f} seconds")
    
    def fit(
        self, 
        X_train: np.ndarray, 
        y_train: np.ndarray, 
        batch_size: int = 32, 
        learning_rate: float = 0.01, 
        epochs: int = 100, 
        X_val: Optional[np.ndarray] = None, 
        y_val: Optional[np.ndarray] = None, 
        verbose: int = 1
    ) -> Dict:
        self.logger.info(f"Training started with {X_train.shape[0]} samples, batch size: {batch_size}, learning rate: {learning_rate}, epochs: {epochs}")
        training_start_time = time.time()
        
        n_samples = X_train.shape[0]
        n_batches = int(np.ceil(n_samples / batch_size))
        
        self.history = {
            'train_loss': [],
            'val_loss': []
        }
        
        for epoch in range(epochs):
            epoch_start_time = time.time()
            self.logger.info(f"Epoch {epoch+1}/{epochs} started")
            
            # Shuffle data
            indices = np.random.permutation(n_samples)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]
            
            batch_losses = []
            batch_times = []
            
            for batch in range(n_batches):
                batch_start_time = time.time()
                
                start_idx = batch * batch_size
                end_idx = min((batch + 1) * batch_size, n_samples)
                actual_batch_size = end_idx - start_idx
                
                self.logger.debug(f"Epoch {epoch+1}, Batch {batch+1}/{n_batches}, Samples: {start_idx}-{end_idx}")
                
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]
                
                # Forward pass
                y_pred = self.forward(X_batch)
                
                # Compute loss
                batch_loss = self._compute_loss(y_pred, y_batch)
                batch_losses.append(batch_loss)
                
                # Backward pass
                self.backward(y_batch)
                
                # Update weights
                self.update_weights(learning_rate)
                
                batch_time = time.time() - batch_start_time
                batch_times.append(batch_time)
                
                if batch % max(1, n_batches // 10) == 0 or batch == n_batches - 1:
                    self.logger.info(f"Epoch {epoch+1}, Batch {batch+1}/{n_batches}: loss = {batch_loss:.6f}, time = {batch_time:.3f}s")
            
            # Calculate epoch metrics
            train_loss = np.mean(batch_losses)
            self.history['train_loss'].append(train_loss)
            
            epoch_time = time.time() - epoch_start_time
            avg_batch_time = np.mean(batch_times)
            
            # Validation
            val_loss = None
            if X_val is not None and y_val is not None:
                val_start_time = time.time()
                self.logger.info(f"Validating on {X_val.shape[0]} samples")
                
                y_val_pred = self.forward(X_val)
                val_loss = self._compute_loss(y_val_pred, y_val)
                self.history['val_loss'].append(val_loss)
                
                val_time = time.time() - val_start_time
                self.logger.info(f"Validation completed in {val_time:.3f}s, loss: {val_loss:.6f}")
            
            # Log epoch summary
            self.logger.info(f"Epoch {epoch+1}/{epochs} completed in {epoch_time:.3f}s - avg batch time: {avg_batch_time:.3f}s - train_loss: {train_loss:.6f}" + 
                           (f" - val_loss: {val_loss:.6f}" if val_loss is not None else ""))
            
            if verbose == 1:
                if val_loss is not None:
                    print(f"Epoch {epoch+1}/{epochs} - loss: {train_loss:.4f} - val_loss: {val_loss:.4f}")
                else:
                    print(f"Epoch {epoch+1}/{epochs} - loss: {train_loss:.4f}")
        
        total_training_time = time.time() - training_start_time
        self.logger.info(f"Training completed in {total_training_time:.3f}s")
        
        return self.history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        self.logger.info(f"Predicting for {X.shape[0]} samples")
        start_time = time.time()
        
        predictions = self.forward(X)
        
        elapsed = time.time() - start_time
        self.logger.info(f"Prediction completed in {elapsed:.3f}s")
        
        return predictions
    
    def display_model(self) -> None:
        pass
    
    def plot_weight_distribution(self, layers: List[int]) -> None:
        pass
    
    def plot_gradient_distribution(self, layers: List[int]) -> None:
        pass
    
    def save_model(self, filename: str) -> None:
       pass
    
    def load_model(self, filename: str) -> None:
        pass

    def plot_training_history(self) -> None:
        pass
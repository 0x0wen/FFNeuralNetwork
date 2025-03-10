import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional, Union, Callable


class FFNN:
    """
    Feedforward Neural Network (FFNN)
    """
    
    def __init__(
        self, 
        layer_sizes: List[int],
        activation_functions: List[str],
        loss_function: str,
        weight_initialization: str,
        weight_init_params: Dict,
    ):
        if len(layer_sizes) < 2:
            raise ValueError("At least 2 layers (input and output) are required")
        
        if len(activation_functions) != len(layer_sizes) - 1:
            raise ValueError("Number of activation functions must be equal to number of layers - 1")
        
        self.layer_sizes = layer_sizes
        self.n_layers = len(layer_sizes)
        self.activation_functions = activation_functions
        self.loss_function = loss_function
        
        self.weights = []
        self.biases = []
        self.gradients_w = []
        self.gradients_b = []
        
        self._initialize_weights(weight_initialization, weight_init_params)
        
        self.z_values = []  
        self.a_values = []  
        
        self.history = {
            'train_loss': [],
            'val_loss': []
        }
    
    def _initialize_weights(self, method: str, params: Dict):
        np.random.seed(params.get('seed', 42))
        
        for i in range(self.n_layers - 1):
            input_size = self.layer_sizes[i]
            output_size = self.layer_sizes[i + 1]
            
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

    
    def _activation_forward(self, Z: np.ndarray, activation: str) -> np.ndarray:
        if activation == "linear":
            return Z
        
        elif activation == "sigmoid":
            return 1 / (1 + np.exp(-np.clip(Z, -500, 500))) 
        
        elif activation == "tanh":
            return np.tanh(Z)
        
        elif activation == "relu":
            return np.maximum(0, Z)
        
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
      pass
    
    def _compute_loss(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        if self.loss_function == "mse":
            return np.mean(np.square(y_pred - y_true))
        
        elif self.loss_function == "binary_crossentropy":
            y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
            return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        
        elif self.loss_function == "categorical_crossentropy":
            y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
            return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
        
        else:
            raise ValueError(f"Unsupported loss function: {self.loss_function}")
    
    def _compute_loss_gradient(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        m = y_true.shape[0]  
        
        if self.loss_function == "mse":
            return (2/m) * (y_pred - y_true)
        
        elif self.loss_function == "binary_crossentropy":
            y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
            return (1/m) * (- y_true / y_pred + (1 - y_true) / (1 - y_pred))
        
        elif self.loss_function == "categorical_crossentropy":
            if self.activation_functions[-1] == "softmax":
                return (1/m) * (y_pred - y_true)
            else:
                y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
                return (1/m) * (- y_true / y_pred)
        
        else:
            raise ValueError(f"Unsupported loss function: {self.loss_function}")
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        self.z_values = []
        self.a_values = [X] 
        
        A = X
        for i in range(self.n_layers - 1):
            W = self.weights[i]
            b = self.biases[i]
            activation = self.activation_functions[i]
            
            Z = np.dot(A, W) + b
            
            A = self._activation_forward(Z, activation)
            
            self.z_values.append(Z)
            self.a_values.append(A)
        
        return A  
    
    def backward(self, y_true: np.ndarray) -> None:
        pass
    
    def update_weights(self, learning_rate: float) -> None:
        pass
    
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
        n_samples = X_train.shape[0]
        n_batches = int(np.ceil(n_samples / batch_size))
        
        self.history = {
            'train_loss': [],
            'val_loss': []
        }
        
        for epoch in range(epochs):
            indices = np.random.permutation(n_samples)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]
            
            batch_losses = []
            for batch in range(n_batches):
                start_idx = batch * batch_size
                end_idx = min((batch + 1) * batch_size, n_samples)
                
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]
                
                y_pred = self.forward(X_batch)
                
                batch_loss = self._compute_loss(y_pred, y_batch)
                batch_losses.append(batch_loss)
                
                self.backward(y_batch)
                
                self.update_weights(learning_rate)
            
            train_loss = np.mean(batch_losses)
            self.history['train_loss'].append(train_loss)
            
            val_loss = None
            if X_val is not None and y_val is not None:
                y_val_pred = self.forward(X_val)
                val_loss = self._compute_loss(y_val_pred, y_val)
                self.history['val_loss'].append(val_loss)
            
            if verbose == 1:
                if val_loss is not None:
                    print(f"Epoch {epoch+1}/{epochs} - loss: {train_loss:.4f} - val_loss: {val_loss:.4f}")
                else:
                    print(f"Epoch {epoch+1}/{epochs} - loss: {train_loss:.4f}")
        
        return self.history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X)
    
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
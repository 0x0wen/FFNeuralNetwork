import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
import time
import argparse

from ffnn import FFNN 

def load_mnist_data():
    print("Loading MNIST dataset...")
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
    
    if y.dtype == object:
        y = y.astype(int)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    encoder = OneHotEncoder(sparse_output=False)
    y_train_encoded = encoder.fit_transform(y_train.reshape(-1, 1))
    y_test_encoded = encoder.transform(y_test.reshape(-1, 1))
    
    print(f"Data loaded. Training samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")
    
    return X_train, X_test, y_train_encoded, y_test_encoded, y_train, y_test

def experiment_width_depth():
    print("\n=== Experiment: Width and Depth ===")
    
    X_train, X_test, y_train, y_test, y_train_orig, y_test_orig = load_mnist_data()
    
    configs = [
        {"name": "Small width", "hidden_layers": [32, 32], "color": "blue"},
        {"name": "Medium width", "hidden_layers": [64, 64], "color": "green"},
        {"name": "Large width", "hidden_layers": [128, 128], "color": "red"},
        
        {"name": "Shallow (1 layer)", "hidden_layers": [64], "color": "purple"},
        {"name": "Medium depth (3 layers)", "hidden_layers": [64, 64, 64], "color": "orange"},
        {"name": "Deep (4 layers)", "hidden_layers": [64, 64, 64, 64], "color": "brown"}
    ]
    
    results = []
    
    plt.figure(figsize=(12, 8))
    
    for config in configs:
        print(f"\nTraining {config['name']} network...")
        
        activations = ['relu'] * len(config['hidden_layers']) + ['softmax']
        
        model = FFNN(
            layer_sizes=[X_train.shape[1]] + config['hidden_layers'] + [y_train.shape[1]],
            activation_functions=activations,
            loss_function='categorical_crossentropy',
            weight_initialization='he'
        )
        
        start_time = time.time()
        history = model.fit(
            X_train=X_train[:5000], 
            y_train=y_train[:5000],
            batch_size=32,
            learning_rate=0.01,
            epochs=20,
            X_val=X_test[:1000],
            y_val=y_test[:1000],
            verbose=1
        )
        train_time = time.time() - start_time
        
        y_pred = model.predict(X_test[:1000])
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_test_classes = np.argmax(y_test[:1000], axis=1)
        
        accuracy = accuracy_score(y_test_classes, y_pred_classes)
        
        results.append({
            "config": config,
            "accuracy": accuracy,
            "train_time": train_time,
            "history": history
        })
        
        plt.plot(history['train_loss'], label=f"{config['name']} (Train)", color=config['color'], linestyle='-')
        plt.plot(history['val_loss'], label=f"{config['name']} (Val)", color=config['color'], linestyle='--')
    
    plt.title('Training and Validation Loss for Different Network Configurations')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('width_depth_loss.png')
    plt.close()
    
    print("\nResults Summary:")
    print("-" * 80)
    print(f"{'Configuration':<20} {'Accuracy':<10} {'Training Time':<15}")
    print("-" * 80)
    
    for result in results:
        config = result['config']
        print(f"{config['name']:<20} {result['accuracy']:.4f}      {result['train_time']:.2f}s")

def experiment_activation_functions():
    print("\n=== Experiment: Activation Functions ===")
    
    X_train, X_test, y_train, y_test, y_train_orig, y_test_orig = load_mnist_data()
    
    activations = ['linear', 'relu', 'sigmoid', 'tanh']
    
    results = []
    
    plt.figure(figsize=(12, 8))
    
    for activation in activations:
        print(f"\nTraining network with {activation} activation...")
        
        model = FFNN(
            layer_sizes=[X_train.shape[1], 64, 32, y_train.shape[1]],
            activation_functions=[activation, activation, 'softmax'],
            loss_function='categorical_crossentropy',
            weight_initialization='he' if activation == 'relu' else 'xavier'
        )
        
        start_time = time.time()
        history = model.fit(
            X_train=X_train[:5000],
            y_train=y_train[:5000],
            batch_size=32,
            learning_rate=0.01,
            epochs=20,
            X_val=X_test[:1000],
            y_val=y_test[:1000],
            verbose=1
        )
        train_time = time.time() - start_time
        
        y_pred = model.predict(X_test[:1000])
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_test_classes = np.argmax(y_test[:1000], axis=1)
        
        accuracy = accuracy_score(y_test_classes, y_pred_classes)
        
        results.append({
            "activation": activation,
            "accuracy": accuracy,
            "train_time": train_time,
            "history": history,
            "model": model
        })
        
        plt.plot(history['train_loss'], label=f"{activation} (Train)")
        plt.plot(history['val_loss'], label=f"{activation} (Val)", linestyle='--')
    
    plt.title('Training and Validation Loss for Different Activation Functions')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('activation_functions_loss.png')
    plt.close()
    
    for result in results:
        activation = result['activation']
        model = result['model']
        
        model.plot_weight_distribution([0, 1])
        plt.suptitle(f'Weight Distributions for {activation} Activation')
        plt.savefig(f'weight_distribution_{activation}.png')
        plt.close()
        
        model.plot_gradient_distribution([0, 1])
        plt.suptitle(f'Gradient Distributions for {activation} Activation')
        plt.savefig(f'gradient_distribution_{activation}.png')
        plt.close()
    
    print("\nResults Summary:")
    print("-" * 80)
    print(f"{'Activation':<10} {'Accuracy':<10} {'Training Time':<15}")
    print("-" * 80)
    
    for result in results:
        print(f"{result['activation']:<10} {result['accuracy']:.4f}      {result['train_time']:.2f}s")

def experiment_learning_rates():
    print("\n=== Experiment: Learning Rates ===")
    
    X_train, X_test, y_train, y_test, y_train_orig, y_test_orig = load_mnist_data()
    
    learning_rates = [0.001, 0.01, 0.1]
    colors = ['blue', 'green', 'red']
    
    results = []
    
    plt.figure(figsize=(12, 8))
    
    for i, lr in enumerate(learning_rates):
        print(f"\nTraining network with learning rate {lr}...")
        
        model = FFNN(
            layer_sizes=[X_train.shape[1], 64, 32, y_train.shape[1]],
            activation_functions=['relu', 'relu', 'softmax'],
            loss_function='categorical_crossentropy',
            weight_initialization='he'
        )
        
        start_time = time.time()
        history = model.fit(
            X_train=X_train[:5000],  
            y_train=y_train[:5000],
            batch_size=32,
            learning_rate=lr,
            epochs=20,
            X_val=X_test[:1000],
            y_val=y_test[:1000],
            verbose=1
        )
        train_time = time.time() - start_time
        
        y_pred = model.predict(X_test[:1000])
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_test_classes = np.argmax(y_test[:1000], axis=1)
        
        accuracy = accuracy_score(y_test_classes, y_pred_classes)
        
        results.append({
            "learning_rate": lr,
            "accuracy": accuracy,
            "train_time": train_time,
            "history": history,
            "model": model
        })
        
        plt.plot(history['train_loss'], label=f"LR={lr} (Train)", color=colors[i], linestyle='-')
        plt.plot(history['val_loss'], label=f"LR={lr} (Val)", color=colors[i], linestyle='--')
    
    plt.title('Training and Validation Loss for Different Learning Rates')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('learning_rates_loss.png')
    plt.close()
    
    for result in results:
        lr = result['learning_rate']
        model = result['model']
        
        model.plot_weight_distribution([0, 1])
        plt.suptitle(f'Weight Distributions for Learning Rate {lr}')
        plt.savefig(f'weight_distribution_lr_{lr}.png')
        plt.close()
        
        model.plot_gradient_distribution([0, 1])
        plt.suptitle(f'Gradient Distributions for Learning Rate {lr}')
        plt.savefig(f'gradient_distribution_lr_{lr}.png')
        plt.close()
    
    print("\nResults Summary:")
    print("-" * 80)
    print(f"{'Learning Rate':<15} {'Accuracy':<10} {'Training Time':<15}")
    print("-" * 80)
    
    for result in results:
        print(f"{result['learning_rate']:<15} {result['accuracy']:.4f}      {result['train_time']:.2f}s")

def experiment_weight_initialization():
    print("\n=== Experiment: Weight Initialization ===")
    
    X_train, X_test, y_train, y_test, y_train_orig, y_test_orig = load_mnist_data()
    
    init_methods = ['zeros', 'random_uniform', 'random_normal', 'xavier', 'he']
    colors = ['blue', 'green', 'red', 'purple', 'orange']
    
    results = []
    
    plt.figure(figsize=(12, 8))
    
    for i, init_method in enumerate(init_methods):
        print(f"\nTraining network with {init_method} initialization...")
        
        if init_method == 'random_uniform':
            init_params = {'lower_bound': -0.1, 'upper_bound': 0.1, 'seed': 42}
        elif init_method == 'random_normal':
            init_params = {'mean': 0, 'variance': 0.1, 'seed': 42}
        else:
            init_params = {'seed': 42}
        
        model = FFNN(
            layer_sizes=[X_train.shape[1], 64, 32, y_train.shape[1]],
            activation_functions=['relu', 'relu', 'softmax'],
            loss_function='categorical_crossentropy',
            weight_initialization=init_method,
            weight_init_params=init_params
        )
        
        start_time = time.time()
        history = model.fit(
            X_train=X_train[:5000], 
            y_train=y_train[:5000],
            batch_size=32,
            learning_rate=0.01,
            epochs=20,
            X_val=X_test[:1000],
            y_val=y_test[:1000],
            verbose=1
        )
        train_time = time.time() - start_time
        
        y_pred = model.predict(X_test[:1000])
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_test_classes = np.argmax(y_test[:1000], axis=1)
        
        accuracy = accuracy_score(y_test_classes, y_pred_classes)
        
        results.append({
            "init_method": init_method,
            "accuracy": accuracy,
            "train_time": train_time,
            "history": history,
            "model": model
        })
        
        # Plot training loss
        plt.plot(history['train_loss'], label=f"{init_method} (Train)", color=colors[i], linestyle='-')
        plt.plot(history['val_loss'], label=f"{init_method} (Val)", color=colors[i], linestyle='--')
    
    plt.title('Training and Validation Loss for Different Weight Initialization Methods')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('weight_initialization_loss.png')
    plt.close()
    
    plt.figure(figsize=(15, 10))
    for i, result in enumerate(results):
        init_method = result['init_method']
        model = result['model']
        
        plt.subplot(2, 3, i+1)
        weights = model.weights[0].flatten()
        plt.hist(weights, bins=50, color=colors[i])
        plt.title(f'Initial Weight Distribution - {init_method}')
        plt.xlabel('Weight Value')
        plt.ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig('initial_weight_distributions.png')
    plt.close()
    
    # Print results summary
    print("\nResults Summary:")
    print("-" * 80)
    print(f"{'Initialization':<15} {'Accuracy':<10} {'Training Time':<15}")
    print("-" * 80)
    
    for result in results:
        print(f"{result['init_method']:<15} {result['accuracy']:.4f}      {result['train_time']:.2f}s")

def compare_with_sklearn():
    print("\n=== Experiment: Comparison with sklearn's MLPClassifier ===")
    
    X_train, X_test, y_train, y_test, y_train_orig, y_test_orig = load_mnist_data()
    
    hidden_layers = [64, 32]
    activation = 'relu'
    learning_rate = 0.01
    max_iter = 20
    batch_size = 32
    
    print("\nTraining our FFNN model...")
    our_model = FFNN(
        layer_sizes=[X_train.shape[1]] + hidden_layers + [y_train.shape[1]],
        activation_functions=[activation, activation, 'softmax'],
        loss_function='categorical_crossentropy',
        weight_initialization='he'
    )
    
    start_time = time.time()
    our_history = our_model.fit(
        X_train=X_train[:5000], 
        y_train=y_train[:5000],
        batch_size=batch_size,
        learning_rate=learning_rate,
        epochs=max_iter,
        X_val=X_test[:1000],
        y_val=y_test[:1000],
        verbose=1
    )
    our_train_time = time.time() - start_time
    
    our_pred = our_model.predict(X_test[:1000])
    our_pred_classes = np.argmax(our_pred, axis=1)
    y_test_classes = np.argmax(y_test[:1000], axis=1)
    
    our_accuracy = accuracy_score(y_test_classes, our_pred_classes)
    
    print("\nTraining sklearn's MLPClassifier...")
    sklearn_model = MLPClassifier(
        hidden_layer_sizes=tuple(hidden_layers),
        activation='relu',
        solver='sgd',
        learning_rate_init=learning_rate,
        max_iter=max_iter,
        batch_size=batch_size,
        random_state=42
    )
    
    start_time = time.time()
    sklearn_model.fit(X_train[:5000], y_train_orig[:5000])
    sklearn_train_time = time.time() - start_time
    
    sklearn_pred = sklearn_model.predict(X_test[:1000])
    sklearn_accuracy = accuracy_score(y_test_orig[:1000], sklearn_pred)
    
    print("\nResults Comparison:")
    print("-" * 80)
    print(f"{'Model':<20} {'Accuracy':<10} {'Training Time':<15}")
    print("-" * 80)
    print(f"{'Our FFNN':<20} {our_accuracy:.4f}      {our_train_time:.2f}s")
    print(f"{'sklearn MLP':<20} {sklearn_accuracy:.4f}      {sklearn_train_time:.2f}s")
    

def main():
    parser = argparse.ArgumentParser(description="Run FFNN experiments")
    parser.add_argument("--experiment", type=str, default="all",
                      help="Experiment to run (width_depth, activation, learning_rate, weight_init, sklearn, all)")
    args = parser.parse_args()
    
    if args.experiment == "width_depth" or args.experiment == "all":
        experiment_width_depth()
    
    if args.experiment == "activation" or args.experiment == "all":
        experiment_activation_functions()
    
    if args.experiment == "learning_rate" or args.experiment == "all":
        experiment_learning_rates()
    
    if args.experiment == "weight_init" or args.experiment == "all":
        experiment_weight_initialization()
    
    if args.experiment == "sklearn" or args.experiment == "all":
        compare_with_sklearn()

if __name__ == "__main__":
    main()
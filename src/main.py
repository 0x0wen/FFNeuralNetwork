import numpy as np
import argparse
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, classification_report

from ffnn import FFNN  # Import FFNN class yang sudah dibuat sebelumnya

def load_data(dataset_name="mnist_784"):
    print(f"Loading {dataset_name} dataset...")

    X, y = fetch_openml(dataset_name, version=1, return_X_y=True, as_frame=False)
    
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

def create_model(input_size, output_size, hidden_layers, activations, loss_function, weight_init):
    layer_sizes = [input_size] + hidden_layers + [output_size]
    
    model = FFNN(
        layer_sizes=layer_sizes,
        activation_functions=activations,
        loss_function=loss_function,
        weight_initialization=weight_init
    )
    
    return model

def main():
    parser = argparse.ArgumentParser(description="Run FFNN model")
    parser.add_argument("--dataset", type=str, default="mnist_784", help="Dataset name")
    parser.add_argument("--hidden_layers", type=str, default="128,64", help="Comma-separated list of hidden layer sizes")
    parser.add_argument("--activations", type=str, default="relu,relu,softmax", help="Comma-separated list of activation functions")
    parser.add_argument("--loss", type=str, default="categorical_crossentropy", help="Loss function")
    parser.add_argument("--weight_init", type=str, default="random_normal", help="Weight initialization method")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--save_model", type=str, default="model.npy", help="Path to save the model")
    args = parser.parse_args()
    
    hidden_layers = [int(size) for size in args.hidden_layers.split(",")]
    activations = args.activations.split(",")
    
    X_train, X_test, y_train_encoded, y_test_encoded, y_train_original, y_test_original = load_data(args.dataset)
    
    input_size = X_train.shape[1]
    output_size = y_train_encoded.shape[1]
    
    model = create_model(
        input_size=input_size,
        output_size=output_size,
        hidden_layers=hidden_layers,
        activations=activations,
        loss_function=args.loss,
        weight_init=args.weight_init
    )
    
    print("\nModel structure:")
    model.display_model()
    
    print("\nTraining model...")
    history = model.fit(
        X_train=X_train,
        y_train=y_train_encoded,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        X_val=X_test,
        y_val=y_test_encoded,
        verbose=1
    )
    
    print("\nEvaluating model...")
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_test_classes = np.argmax(y_test_encoded, axis=1)
    
    accuracy = accuracy_score(y_test_classes, y_pred_classes)
    print(f"Test accuracy: {accuracy:.4f}")
    
    print("\nClassification report:")
    print(classification_report(y_test_classes, y_pred_classes))
    
    if args.save_model:
        print(f"\nSaving model to {args.save_model}...")
        model.save_model(args.save_model)
    
    print("\nPlotting weight distributions...")
    model.plot_weight_distribution(list(range(len(model.weights))))
    
    print("\nPlotting gradient distributions...")
    model.plot_gradient_distribution(list(range(len(model.gradients_w))))
    
    print("\nPlotting training history...")
    model.plot_training_history()

if __name__ == "__main__":
    main()
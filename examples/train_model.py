import numpy as np
import os
import sys

# Add the parent directory to the path so we can import the src package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.neural_network import NeuralNetwork
from src.functions import sigmoid, meansquare, adam

"""
Example script for initializing and training a neural network on image data.
This script demonstrates how to use the neural network implementation for image classification.
"""

def main():
    """
    Main function to train the neural network.
    """
    # Network architecture: [input_size, hidden_layer1, hidden_layer2, output_size]
    network_architecture = [100*100, 50, 30, 5]
    
    # Initialize the neural network
    print("Initializing neural network...")
    nn = NeuralNetwork(
        network_architecture,
        sigmoid,
        meansquare,
        adam,
        0.05
    )
    
    # Load training data
    print("Loading training data...")
    try:
        # Update paths to use relative paths or environment variables
        data_dir = os.getenv('DATA_DIR', '.')
        
        X_train = np.load(os.path.join(data_dir, "train_image100pdroite.npy")).astype(float) / 255
        y_train = np.load(os.path.join(data_dir, "train_labels.npy")).astype(float)
        
        # Reshape input data
        X_train = np.reshape(X_train, (400, 100*100, 1))
        
        # Load test data
        X_test = np.load(os.path.join(data_dir, "test_image100pdroite.npy")).astype(float) / 255
        y_test = np.load(os.path.join(data_dir, "test_labels.npy")).astype(float)
        
        # Reshape test data
        X_test = np.reshape(X_test, (100, 100*100, 1))
        
        print(f"Loaded {X_train.shape[0]} training samples and {X_test.shape[0]} test samples")
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        print("Please ensure the data files are in the correct location or set the DATA_DIR environment variable.")
        return
    
    # Train the network
    print("Training neural network...")
    error_train, error_test = nn.train_set(
        epoch=400,
        gamma=10**(-8),
        input_data=X_train,
        target_output=y_train,
        test_input=X_test,
        test_output=y_test,
        error_evaluate=True
    )
    
    # Save the model and error rates
    print("Saving model and results...")
    nn.save_model('trained_model.pkl')
    np.save('error_train', error_train)
    np.save('error_test', error_test)
    
    print("Training complete!")
    
    # Uncomment to determine optimal learning rate
    # print("Determining optimal learning rate...")
    # nn.lr_determine(10**(-8), X_train, y_train)

if __name__ == "__main__":
    main()
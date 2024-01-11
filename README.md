# NeuralNetwork_XOR_Backprop
This script demonstrates the implementation of a simple neural network using the backpropagation algorithm for the XOR problem. The network has a 3-4-1 architecture, consisting of an input layer with 3 neurons, a hidden layer with 4 neurons, and an output layer with 1 neuron.


- Initialization:
Weight matrices W1 and W2 are initialized with random values between -1 and 1.
- Backpropagation Algorithm:
The backprop function implements the backpropagation algorithm for training the neural network.
The algorithm includes forward propagation to compute the output and backward propagation to adjust the weights based on the error.
- Training:
The network is trained with XOR data, where X represents the input and D represents the correct output.
The weights are updated iteratively for 10,000 epochs.
- Testing:
The trained network is tested with the same XOR data to observe the output.
The final weights (W1 and W2) are printed.
- Note:
The sigmoid activation function is used in the network.
The script uses random initialization and a fixed learning rate (alpha).
The XOR data and the resulting output for each input are printed during testing.

# NeuralNetwork_using_TensorFlow&Keras

This script demonstrates the training and evaluation of four different neural network models for image classification on the CIFAR-10 dataset using TensorFlow and Keras. Here's a breakdown of each model:

- Model 1:
    - Architecture:
    Input layer: Flatten layer for reshaping the input.
    Hidden layers: Three dense layers with ReLU activation functions.
    Output layer: Dense layer with softmax activation for multiclass classification.
    -  Compilation:
    Loss function: Sparse categorical crossentropy.
    Optimizer: Stochastic Gradient Descent (SGD).
    Metrics: Accuracy.
    - Training:
    Trained for 10 epochs on the training set.
    - Evaluation:
    Predictions are compared with actual labels for the first 20 test instances.
- Model 2:
    - Architecture:
    Similar to Model 1 but with increased neurons in the first dense layer.
    Output layer uses the sigmoid activation function.
    - Compilation:
    Loss function: Sparse categorical crossentropy.
    Optimizer: Adam optimizer with a custom learning rate and epsilon value.
    Metrics: Accuracy.
    - Training:
    Trained for 20 epochs on the training set.
- Model 3:
    - Architecture:
    Similar to Model 2 but with different activation functions (selu for hidden layers, sigmoid for the output layer).
    - Compilation:
    Loss function: Sparse categorical crossentropy.
    Optimizer: Stochastic Gradient Descent (SGD) with a custom learning rate and momentum.
    Metrics: Accuracy.
    - Training:
    Trained for 25 epochs on the training set.
- Model 4:
    - Architecture:
    Similar to Model 2 but with a different optimizer (Nadam) and different layer sizes.
    - Compilation:
    Loss function: Sparse categorical crossentropy.
    Optimizer: Nadam optimizer with custom parameters.
    Metrics: Accuracy.
    - Training:
    Trained for 20 epochs on the training set.
- Note:
Each model is trained and evaluated separately.
The choice of activation functions, optimizers, and other hyperparameters can significantly impact model performance.
The script prints the actual and predicted classes for the first 20 test instances for each model, allowing comparison.



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
# Note:
The sigmoid activation function is used in the network.
The script uses random initialization and a fixed learning rate (alpha).
The XOR data and the resulting output for each input are printed during testing.

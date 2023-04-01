import random

# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + pow(2.718, -x))

# Forward propagation function
def forward_propagation(input_data, weights):
    hidden_layer_activation = 0
    for i in range(len(input_data)):
        hidden_layer_activation += input_data[i] * weights[0][i]
    hidden_layer_output = sigmoid(hidden_layer_activation)
    
    output_layer_activation = hidden_layer_output * weights[1][0]
    output = sigmoid(output_layer_activation)
    
    return output

# Initialize the weights
def initialize_weights(input_dim):
    weights = []
    weights.append([random.uniform(-1, 1) for i in range(input_dim)])
    weights.append([random.uniform(-1, 1)])
    return weights

# Train the neural network
def train(X, y, epochs, lr):
    weights = initialize_weights(1)
    for i in range(epochs):
        for j in range(len(X)):
            input_data = X[j]
            target_output = y[j]
            
            # Forward propagation
            output = forward_propagation(input_data, weights)
            
            # Backward propagation
            error = target_output - output
            output_error = error * output * (1 - output)
            hidden_error = output_error * weights[1][0] * sigmoid(input_data[0]) * (1 - sigmoid(input_data[0]))
            
            # Update weights
            weights[1][0] += lr * output_error * sigmoid(hidden_error)
            for k in range(len(input_data)):
                weights[0][k] += lr * hidden_error * input_data[k]
                
    return weights

# Predict pass/fail
def predict(X, weights):
    predictions = []
    for i in range(len(X)):
        input_data = X[i]
        output = forward_propagation(input_data, weights)
        predictions.append(int(round(output)))
    return predictions

# Input data
X = [[50], [60], [70], [80], [90], [95], [100]]
y = [0, 0, 1, 1, 1, 1, 1]

# Train the neural network
weights = train(X, y, epochs=1000, lr=0.1)

# Predict pass/fail
new_X = [[55], [65], [75], [85], [95], [100], [40]]
predictions = predict(new_X, weights)
print(predictions)

class Perceptron:
    def __init__(self, num_inputs):
        self.weights = [0] * num_inputs
        self.bias = 0

    def predict(self, inputs):
        activation = self.bias
        for i in range(len(inputs)):
            activation += inputs[i] * self.weights[i]
        return 1 if activation >= 0 else 0

    def train(self, training_inputs, labels, num_epochs):
        for epoch in range(num_epochs):
            for inputs, label in zip(training_inputs, labels):
                prediction = self.predict(inputs)
                error = label - prediction
                self.bias += error
                for i in range(len(inputs)):
                    self.weights[i] += error * inputs[i]

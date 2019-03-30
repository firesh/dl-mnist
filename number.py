import numpy
import time
import scipy.special
import matplotlib.pyplot as plt

class neuralNetwork:
    # init the nenual network
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        self.lr = learningrate

        # init the weight
        self.wih = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        self.who = numpy.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))

        pass
    
    # active function
    def active_function(self, x):
        return scipy.special.expit(x)

    # train the nenual network
    def train(self, inputs_list, targets_list):
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T

        # output
        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.active_function(hidden_inputs)
        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.active_function(final_inputs)

        # error
        output_errors = targets - final_outputs
        hidden_errors = numpy.dot(self.who.T, output_errors)

        # update the weight
        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)), numpy.transpose(hidden_outputs))
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), numpy.transpose(inputs))

        pass
    
    # query the nenual network
    def query(self, inputs_list):
        inputs = numpy.array(inputs_list, ndmin=2).T

        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.active_function(hidden_inputs)
        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.active_function(final_inputs)

        return final_outputs

input_nodes = 784
hidden_nodes = 300
output_nodes = 10
learning_rate = 0.1
epochs = 10

n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

# load the mnist train data
train_data_file = open("dataset/mnist_train.csv", 'r')
train_data_list = train_data_file.readlines()
train_data_file.close()

# train the neural network
for e in range(epochs):
    for record in train_data_list:
        all_values = record.split(',')
        inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        targets = numpy.zeros(output_nodes) + 0.01
        targets[int(all_values[0])] = 0.99
        n.train(inputs, targets)
        pass

# load the mnist test data
test_data_file = open("dataset/mnist_test.csv", 'r')
test_data_list = test_data_file.readlines()
test_data_file.close()

# test the trained neural network
scorecard = []
for record in test_data_list:
    all_values = record.split(',')
    expected_value = int(all_values[0])

    # image_array = numpy.asfarray(all_values[1:]).reshape((28, 28))
    # plt.imshow(image_array, cmap='Greys', interpolation='None')
    # plt.show()

    outputValue = numpy.argmax(n.query((numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01))
    if expected_value == outputValue:
        scorecard.append(1)
    else:
        scorecard.append(0)
    
scorecard_array = numpy.asarray(scorecard)
performance = scorecard_array.sum() / scorecard_array.size
print('performance:', performance)
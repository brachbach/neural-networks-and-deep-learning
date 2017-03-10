import mnist_loader
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
import network

# [input layer, hidden layers, output layer]
net = network.Network([784, 2, 3, 10])

# (...epochs,  mini-batch size, learning rate, .. )

net.SGD(training_data, 30, 10, 3.0, test_data)
import mnist_loader
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
import network2

# [input layer, hidden layers, output layer]
net = network2.Network([784, 30, 10], cost=network2.CrossEntropyCost)
net.large_weight_initializer()

# (...epochs,  mini-batch size, learning rate, .. )
net.SGD(training_data, 30, 10, 1, lmbda=10, evaluation_data=test_data, monitor_evaluation_accuracy=True)
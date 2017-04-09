import network3

from network3 import Network
from network3 import ConvPoolLayer, FullyConnectedLayer, SoftmaxLayer, ReLU

training_data, validation_data, test_data = network3.load_data_shared()
mini_batch_size = 10

net = Network([
        ConvPoolLayer(image_shape=(mini_batch_size, 1, 28, 28), 
                      filter_shape=(5, 1, 5, 5), 
                      poolsize=(2, 2)),
        FullyConnectedLayer(n_in=5*12*12, n_out=20),
        SoftmaxLayer(n_in=20, n_out=10)], mini_batch_size)

net.SGD(training_data, 1, mini_batch_size, 0.1, 
            validation_data, test_data)

net.run_on_data(test_data)
"""This tutorial introduces the LeNet5 neural network architecture
using Theano.  LeNet5 is a convolutional neural network, good for
classifying images. This tutorial shows how to build the architecture,
and comes with all the hyper-parameters you need to reproduce the
paper's MNIST results.


This implementation simplifies the model in the following ways:

 - LeNetConvPool doesn't implement location-specific gain and bias parameters
 - LeNetConvPool doesn't implement pooling by average, it implements pooling
   by max.
 - Digit classification is implemented with a logistic regression rather than
   an RBF network
 - LeNet5 was not fully-connected convolutions at second layer

References:
 - Y. LeCun, L. Bottou, Y. Bengio and P. Haffner:
   Gradient-Based Learning Applied to Document
   Recognition, Proceedings of the IEEE, 86(11):2278-2324, November 1998.
   http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf

"""

import os
import sys
import timeit

import pickle
import numpy

import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv2d

from logistic_sgd import load_data
from train import ol_data
from mlp import HiddenLayer

numpy.set_printoptions(threshold=numpy.nan)

class LeNetConvPoolLayer:
    """Pool Layer of a convolutional network """

    def __init__(self, rng, input, filter_shape, image_shape, poolsize=(2, 2)):
        """
        Allocate a LeNetConvPoolLayer with shared variable internal parameters.

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height, filter width)

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows, #cols)
        """

        assert image_shape[1] == filter_shape[1]
        self.input = input

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = numpy.prod(filter_shape[1:])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) //
                   numpy.prod(poolsize))
        # initialize weights with random weights
        W_bound = numpy.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared(
            numpy.asarray(
                rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                dtype=theano.config.floatX
            ),
            borrow=True
        )

        # the bias is a 1D tensor -- one bias per output feature map
        b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)

        # convolve input feature maps with filters
        conv_out = conv2d(
            input=input,
            filters=self.W,
            filter_shape=filter_shape,
            border_mode='half',
            input_shape=image_shape
        )

        # downsample each feature map individually, using maxpooling
        # pooled_out = downsample.max_pool_2d(
#             input=conv_out,
#             ds=poolsize,
#             ignore_border=True
#             )
        pooled_out = conv_out

        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1, n_filters, 1, 1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        # store parameters of this layer
        self.params = [self.W, self.b]

        # keep track of model input
        self.input = input


class ConvNetwork:
    def __init__(self, nkerns=[100, 100, 50], batch_size=20):
        self.rng = numpy.random.RandomState(23455)

        self.batch_size = batch_size

        # allocate symbolic variables for the data
        self.index = T.lscalar()  # index to a [mini]batch

        # start-snippet-1
        self.x = T.matrix('x')   # the data is presented as rasterized images
        self.y = T.ivector('y')  # the labels are presented as 1D vector of
    
        print('... building the model')

        self.layer0_input = self.x.reshape((self.batch_size, 2, 15, 15))

        self.layer0 = LeNetConvPoolLayer(
            self.rng,
            input=self.layer0_input,
            image_shape=(self.batch_size, 2, 15, 15),
            filter_shape=(nkerns[0], 2, 11, 11),
            poolsize=(1, 1))

        self.layer1 = LeNetConvPoolLayer(
            self.rng,
            input=self.layer0.output,
            image_shape=(self.batch_size, nkerns[0], 15, 15),
            filter_shape=(nkerns[1], nkerns[0], 7, 7),
            poolsize=(1, 1))

        self.layer2 = LeNetConvPoolLayer(
            self.rng,
            input=self.layer1.output,
            image_shape=(self.batch_size, nkerns[1], 15, 15),
            filter_shape=(nkerns[2], nkerns[1], 5, 5),
            poolsize=(1, 1))

        self.layer3 = LeNetConvPoolLayer(
            self.rng,
            input=self.layer2.output,
            image_shape=(self.batch_size, nkerns[2], 15, 15),
            filter_shape=(1, nkerns[2], 5, 5),
            poolsize=(1, 1))
    
        self.layer4 = LeNetConvPoolLayer(
            self.rng,
            input=self.layer3.output,
            image_shape=(self.batch_size, nkerns[3], 15, 15),
            filter_shape=(1, nkerns[3], 1, 1),
            poolsize=(1, 1))

        self.layer4_output = T.nnet.softmax(self.layer4.output.flatten(2))
        self.params = self.layer4.params + self.layer3.params + self.layer2.params + self.layer1.params + self.layer0.params

    def train(self, train_sets, valid_sets, test_sets, 
              n_epochs=200, learning_rate=0.1):

        train_set_x, train_set_y = train_sets
        valid_set_x, valid_set_y = valid_sets
        test_set_x, test_set_y = test_sets

        n_train_batches = train_set_x.get_value(borrow=True).shape[0]
        n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
        n_test_batches = test_set_x.get_value(borrow=True).shape[0]
        n_train_batches //= self.batch_size
        n_valid_batches //= self.batch_size
        n_test_batches //= self.batch_size

        cost = -T.mean(T.log(
                self.layer4_output[T.arange(self.y.shape[0]), self.y]))
        error = T.mean(T.neq(T.argmax(self.layer4_output, axis=1), self.y))

        params = self.params
        grads = T.grad(cost, params)
        updates = [
            (param_i, param_i - learning_rate * grad_i)
            for param_i, grad_i in zip(params, grads)
            ]

        index = self.index
        batch_size = self.batch_size
        x = self.x
        y = self.y

        test_model = theano.function(
            [index],
            error,
            givens={
                x: test_set_x[index * batch_size: (index + 1) * batch_size],
                y: test_set_y[index * batch_size: (index + 1) * batch_size]
                }
            )

        validate_model = theano.function(
            [index],
            error,
            givens={
                x: valid_set_x[index * batch_size: (index + 1) * batch_size],
                y: valid_set_y[index * batch_size: (index + 1) * batch_size]
                }
            )
        
        train_model = theano.function(
            [index],
            cost,
            updates=updates,
            givens={
                x: train_set_x[index * batch_size: (index + 1) * batch_size],
                y: train_set_y[index * batch_size: (index + 1) * batch_size]
                }
            )
        
        print('... training')
        # early-stopping parameters
        patience = 10000  # look as this many examples regardless
        patience_increase = 2  # wait this much longer when a new best is found
        improvement_threshold = 0.995
        validation_frequency = min(n_train_batches, patience // 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

        best_validation_loss = numpy.inf
        best_iter = 0
        test_score = 0.
        start_time = timeit.default_timer()
        
        epoch = 0
        done_looping = False

        while (epoch < n_epochs):
            epoch = epoch + 1
            for minibatch_index in range(n_train_batches):
                
                iter = (epoch - 1) * n_train_batches + minibatch_index
            
                if iter % 100 == 0:
                    print('training @ iter = ', iter)
                cost_ij = train_model(minibatch_index)

                if (iter + 1) % validation_frequency == 0:

                    # compute zero-one loss on validation set
                    validation_losses = [validate_model(i) for i
                                         in range(n_valid_batches)]
                    this_validation_loss = numpy.mean(validation_losses)
                    print('epoch {}, minibatch {}/{}, validation error {}%'.format(
                            epoch, minibatch_index + 1, n_train_batches,
                            this_validation_loss * 100.))
                    with open('model_{}.mod'.format(iter), 'wb') as f:
                        pickle.dump(self.dump(), f)
                    # if we got the best validation score until now
                    if this_validation_loss < best_validation_loss:

                    
                        if this_validation_loss < best_validation_loss *  \
                                improvement_threshold:
                            patience = max(patience, iter * patience_increase)

                        # save best validation score and iteration number
                            best_validation_loss = this_validation_loss
                            best_iter = iter

                    test_losses = [
                        test_model(i)
                        for i in range(n_test_batches)
                    ]
                    test_score = numpy.mean(test_losses)
                    print(('     epoch {}, minibatch {}/{}, test error of '
                           'best model {}%').format(
                            epoch, minibatch_index + 1, n_train_batches,
                            test_score * 100.))
                    with open('test_{}.res'.format(iter), 'w') as f:
                        print(network.predict(test_set_x), file=f)

            
        end_time = timeit.default_timer()
        print('Optimization complete.')
        print('Best validation score of %f %% obtained at iteration %i, '
              'with test performance %f %%' %
              (best_validation_loss * 100., best_iter + 1, test_score * 100.))
        print(('The code for file ' +
               os.path.split(__file__)[1] +
               ' ran for %.2fm' % ((end_time - start_time) / 60.)), file=sys.stderr)

    def predict(self, data):
        set_x = data
        n_batches = set_x.get_value(borrow=True).shape[0]
        n_batches //= self.batch_size

        prediction = self.layer3_output

        index = self.index
        x = self.x
        batch_size = self.batch_size

        pred = theano.function(
            [index],
            prediction,
            givens={
                x: set_x[index * batch_size: (index + 1) * batch_size],
                }
            )
        pred_list = numpy.array([p for i in range(n_batches) for p in pred(i)])
        return pred_list

    def dump(self):
        return [i.get_value() for i in self.params]

    def load(self, values):
        for p, value in zip(self.params, values):
            p.set_value(value)

if __name__ == '__main__':
    network = ConvNetwork()
    datasets = ol_data('../data/games.xml')
    network.train(datasets[0], datasets[1], datasets[2], n_epochs=5)
#     with open('trained.mod', 'rb') as savefile:
#         network.load(pickle.load(savefile))
#     p = network.predict(datasets[2][0])
#     print(p)
    

def experiment(state, channel):
    evaluate_lenet5(state.learning_rate, dataset=state.dataset)

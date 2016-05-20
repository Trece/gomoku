import os
import sys
import timeit

import pickle
import numpy

import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv2d

from logistic_sgd import load_data, LinearNetwork
from train import ol_win_data
from mlp import HiddenLayer
from convolutional_mlp import LeNetConvPoolLayer

# For debugging convenience set to print all numbers in the numpy array
numpy.set_printoptions(threshold=numpy.nan)

class ConvNetwork:
    def __init__(self, nkerns=[100, 100, 100, 1], batch_size=2):
        '''
        nkerns: an array representing how many filters each layer has
        batch_size: a integer indicates batch size
        '''

        self.rng = numpy.random.RandomState(23455)
        self.batch_size = batch_size

        # allocate symbolic variables for the data
        self.index = T.lscalar()  # index to a [mini]batch

        # start-snippet-1
        self.x = T.matrix('x')   # the data is presented as rasterized images
        self.y = T.ivector('y')  # the labels are presented as 1D vector of
    
        print('... building the model')

        '''
        The actual input size consists two matrix of board size, 
        so it's 2 * 15 * 15
        Before feeding intput into the next layer, they are padded 
        such that All layers' outputs have exact size 15*15
        '''
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
            filter_shape=(nkerns[3], nkerns[2], 5, 5),
            poolsize=(1, 1))


        layer4_input = self.layer3.output.flatten(2)

        self.layer4 = HiddenLayer(
            self.rng,
            input=layer4_input,
            n_in=nkerns[3]*15*15,
            n_out=128)
        
        self.layer5 = LinearNetwork(
            self.layer4.output,
            128)
        
        self.final_output = self.layer5.output

        # add up all the parameters
        self.params = (self.layer5.params + self.layer4.params
                       + self.layer3.params + self.layer2.params
                       + self.layer1.params + self.layer0.params)

    def train(self, train_sets, valid_sets, test_sets, 
              n_epochs=200, learning_rate=0.5):

        train_set_x, train_set_y = train_sets
        valid_set_x, valid_set_y = valid_sets
        test_set_x, test_set_y = test_sets

        n_train_batches = train_set_x.get_value(borrow=True).shape[0]
        n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
        n_test_batches = test_set_x.get_value(borrow=True).shape[0]
        n_train_batches //= self.batch_size
        n_valid_batches //= self.batch_size
        n_test_batches //= self.batch_size

        cost = T.mean((self.final_output-self.y) ** 2)
        error = cost

        # find all the parameters and update them using gradient descent
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
        
        layer2_output = theano.function(
            [index],
            self.layer2.output,
            givens={
                x:train_set_x[index * batch_size: (index + 1) * batch_size]
                }
            )

        prediction = theano.function(
            [index],
            self.final_output,
            givens={
                x:train_set_x[index * batch_size: (index + 1) * batch_size]
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
            training_loss = []
            for minibatch_index in range(n_train_batches):
                
                iter = (epoch - 1) * n_train_batches + minibatch_index
                
                cost_ij = train_model(minibatch_index)
                training_loss.append(cost_ij)
                if iter % 100 == 0:
                    print('training @ iter = ', iter)
                    print('cost = {}'.format(cost_ij))
                    print('b = {}'.format(layer2_output(minibatch_index)))
                    print('prediction is {}'.format(prediction(minibatch_index)))
                    print('', flush=True)
                if (iter + 1) % validation_frequency == 0:

                    # compute zero-one loss on validation set
                    validation_losses = [validate_model(i) for i
                                         in range(n_valid_batches)]
                    this_validation_loss = numpy.mean(validation_losses)
                    this_training_loss = numpy.mean(numpy.array(training_loss))
                    print('epoch {}, minibatch {}/{}, validation MSE {}, training MSE{}'.format(
                            epoch, minibatch_index + 1, n_train_batches,
                            this_validation_loss, this_training_loss))
                    training_loss = []
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
                           'best model MSE {}').format(
                            epoch, minibatch_index + 1, n_train_batches,
                            test_score))
                    with open('test_{}.res'.format(iter), 'w') as f:
                        print(network.predict(test_set_x), file=f)

            
        end_time = timeit.default_timer()
        print('Optimization complete.')
        print('Best validation MSE %f obtained at iteration %i, '
              'with test MSE %f' %
              (best_validation_loss, best_iter + 1, test_score))
        print(('The code for file ' +
               os.path.split(__file__)[1] +
               ' ran for %.2fm' % ((end_time - start_time) / 60.)), file=sys.stderr)

    def predict(self, data):
        set_x = data
        n_batches = set_x.get_value(borrow=True).shape[0]
        n_batches //= self.batch_size

        prediction = self.final_output

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
        pred_list = numpy.array([pred(i) for i in range(n_batches)])
        return pred_list

    def dump(self):
        return [i.get_value() for i in self.params]

    def load(self, values):
        for p, value in zip(self.params, values):
            p.set_value(value)

if __name__ == '__main__':
    network = ConvNetwork()
    if len(sys.argv) > 1:
        filename = sys.argv[1]
        params = pickle.load(open(filename, 'rb'))
        network.load(params)
        print('using {} as a start'.format(filename))
    datasets = ol_win_data('../data/games.xml')
    network.train(datasets[0], datasets[1], datasets[2], n_epochs=1)
#     with open('trained.mod', 'rb') as savefile:
#         network.load(pickle.load(savefile))
#     p = network.predict(datasets[2][0])
#     print(p)

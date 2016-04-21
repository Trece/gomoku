import xml.etree.ElementTree as ET
from copy import deepcopy
from pprint import pprint

import numpy
import theano
import theano.tensor as T


DEBUG = True
OPENING_N = 3

def moveindex(move):
    row = ord(move[0]) - ord('a')
    col = int(move[1:]) - 1
    return row, col

def game2img(game):
    moves = [moveindex(move) for move in game.split()]
    turn = 0
    board_black = [[0 for i in range(15)] for j in range(15)]
    board_white = [[0 for i in range(15)] for j in range(15)]
    board = [board_black, board_white]
    b_data = []
    m_data = []
    for move in moves[:OPENING_N]:
        i, j = move
        if board[turn][i][j] != 0:
            print('error')
        board[turn][i][j] = 1
        turn = 1 - turn
    for move in moves[OPENING_N:]:
        b_data.append(deepcopy([board[turn], board[1-turn]]))
        m_data.append(move)
        i, j = move
        if board[turn][i][j] != 0:
            print('error')
        board[turn][i][j] = 1
        turn = 1 - turn
    return b_data, m_data
        

def ol_data(filename):
    '''
    input: string
    output: a list of tuples per yield, each tuple represents a move
    '''
    tree = ET.parse(filename)
    root = tree.getroot()
    if DEBUG:
        root = root[:50]
    data = [None for g in root]
    for i, game in enumerate(root):
        board_string = game.find('board').text
        data[i] = game2img(board_string)
    data_x = [x for d in data for x in d[0]]
    data_y = [y for d in data for y in d[1]]

    def shared_dataset(data_xy, borrow=True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy
        shared_x = theano.shared(numpy.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue
        return shared_x, T.cast(shared_y, 'int32')

    n_validate = 200
    n_test = 200
    n_train = len(data_x) - n_validate - n_test
    n_validate = n_train + n_validate
    n_test = n_validate + n_test
    
    train_x, train_y = shared_dataset((data_x[:n_train],
                                       data_y[:n_train]))
    validate_x, validate_y = shared_dataset((data_x[n_train:n_validate],
                                             data_y[n_train:n_validate]))
    test_x, text_y = shared_dataset((data_x[n_validate:n_test],
                                     data_y[n_validate:n_test]))
    
    print('shape of x is {}'.format(train_x.get_value(borrow=True).shape))

if __name__ == '__main__':
    ol_data('../data/games.xml')

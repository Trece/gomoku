import xml.etree.ElementTree as ET
from copy import deepcopy
from pprint import pprint
from random import randrange

import pickle

import numpy
import theano
import theano.tensor as T


DEBUG = True
OPENING_N = 3

def moveindex(move):
    row = ord(move[0]) - ord('a')
    col = int(move[1:]) - 1
    return row, col

class Board:
    def __init__(self):
        self.black_board = [[0 for i in range(15)] for j in range(15)]
        self.white_board = [[0 for i in range(15)] for j in range(15)]

def check_n(board, pos, n):
    diff = [[1, 1], [0, 1], [1, -1], [1, 0]] 
    i0 = pos[0]
    j0 = pos[1]
    if board[i0, j0] == 1:
        return False
    for s in range(4):
        id, jd = diff[s]
        plus = 0
        minus = 0
        i, j = i0, j0 
        for step in range(n):
            i += id
            j += jd
            if i >= 15 or j >= 15 or i < 0 or j < 0 or board[i, j] == 0:
                break
            else:
                plus += 1
        i, j = i0, j0
        for step in range(n):
            i -= id
            j -= jd
            if i >= 15 or j >= 15 or i < 0 or j < 0 or board[i, j] == 0:
                break
            else:
                minus += 1
        if plus + minus + 1 >= n:
            return True
        
    return False


def game2img(game):
    '''
    input: a string that records the game, moves separated by spaces
    output: an array of matrices that represents the features, 
            and the array of actual next moves
    '''
    moves = [moveindex(move) for move in game.split()]
    turn = 0
    board_black = numpy.zeros((15, 15), dtype='int32')
    board_white = numpy.zeros((15, 15), dtype='int32')
    
    board = numpy.array([board_black, board_white])
    
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

    return numpy.array(b_data), m_data

def game_pos(game):
    '''
    input: a string that records the game, moves separated by spaces
    output: a random situation during the game
    '''
    moves = [moveindex(move) for move in game.split()]
    board_black = numpy.zeros((15, 15), dtype='int32')
    board_white = numpy.zeros((15, 15), dtype='int32')
    board = [board_black, board_white]
    num = None
    turn = 0
    if len(moves) > OPENING_N:
        num = randrange(OPENING_N, len(moves)) 
    else:
        num = OPENING_N - 1
    for move in moves[:num]:
        i, j = move
        if board[turn][i][j] != 0:
            print('error')
        board[turn][i][j] = 1
        turn = 1 - turn
    return numpy.array(board)

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

def ol_move_data(filename):
    '''
    input: string
    output: a list of tuples per yield, each tuple represents a move
    '''
    tree = ET.parse(filename)
    root = tree.getroot()
    if DEBUG:
        root = root[:50000]
    data = []
    for i, game in enumerate(root):
        board_string = game.find('board').text
        if not board_string or'--' in board_string:
            pass
        else:
            data.append(game2img(board_string))
        if i % 50 == 0:
            print('no{}'.format(i))
    print("total data: {}".format(len(data)))
    # print(data)
    data_x = [x.reshape(2*15*15) for d in data for x in d[0]]
    data_y = [y[0]*15 + y[1] for d in data for y in d[1]]


    n_validate = 10000
    n_test = 400
    n_train = len(data_x) - n_validate - n_test
    n_validate = n_train + n_validate
    n_test = n_validate + n_test
    
    print('total train data: {}'.format(n_train))
    with open('test_results', 'w') as f:
        print(data_y[n_validate:n_test], file=f)
    test_xy = (data_x[n_validate:n_test], data_y[n_validate:n_test])
    with open('test_case.pkl', 'wb') as f:
        pickle.dump(test_xy, f)
    train_x, train_y = shared_dataset((data_x[:n_train],
                                       data_y[:n_train]))
    validate_x, validate_y = shared_dataset((data_x[n_train:n_validate],
                                             data_y[n_train:n_validate]))
    test_x, test_y = shared_dataset((data_x[n_validate:n_test],
                                     data_y[n_validate:n_test]))
    rval = [(train_x, train_y), (validate_x, validate_y), (test_x, test_y)]
    return rval

def ol_win_data(filename):
    '''
    input: string
    output: a random position of the board
    '''
    tree = ET.parse(filename)
    root = tree.getroot()
    if DEBUG:
        root = root[:1000]
    data = []
    for i, game in enumerate(root):
        if i % 50 == 0:
            print('no{}'.format(i))
        board_string = game.find('board').text
        if not board_string or'--' in board_string:
            continue
        win_side = game.find('winner').text
        reason = game.find('winby').text
        if not reason in ['resign', 'five'] or not win_side in ['black', 'white']:
            continue
        win_side = (1 if win_side == 'black' else -1)
        data.append((game_pos(board_string), win_side))
    print("total data: {}".format(len(data)))
    data_x = [d[0].reshape(2*15*15) for d in data]
    data_y = [d[1] for d in data]
    print(data_y)

    n_validate = 100
    n_test = 40
    n_train = len(data_x) - n_validate - n_test
    n_validate = n_train + n_validate
    n_test = n_validate + n_test
    
    print('total train data: {}'.format(n_train))
    with open('test_results', 'w') as f:
        print(data_y[n_validate:n_test], file=f)
    test_xy = (data_x[n_validate:n_test], data_y[n_validate:n_test])
    with open('test_case.pkl', 'wb') as f:
        pickle.dump(test_xy, f)
    train_x, train_y = shared_dataset((data_x[:n_train],
                                       data_y[:n_train]))
    validate_x, validate_y = shared_dataset((data_x[n_train:n_validate],
                                             data_y[n_train:n_validate]))
    test_x, test_y = shared_dataset((data_x[n_validate:n_test],
                                     data_y[n_validate:n_test]))
    rval = [(train_x, train_y), (validate_x, validate_y), (test_x, test_y)]
    return rval


if __name__ == '__main__':
    numpy.set_printoptions(threshold=numpy.nan)
    print(ol_win_data('../data/games.xml'))
#     a = numpy.zeros(15*15).reshape((15, 15))
#     a[[4, 5, 6]] = 1
#     b = []
#     for i in range(15):
#         for j in range(15):
#             b.append(1 if check_n(a, (i, j), 5) else 0)
#     x = numpy.array(b).reshape((15, 15))
#     print(x)
    

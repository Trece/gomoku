import xml.etree.ElementTree as ET
from copy import deepcopy
from pprint import pprint
from random import randrange, shuffle

import pickle

import numpy
import theano
import theano.tensor as T


DEBUG = True
OPENING_N = 3

def moveindex(move, direction=0):
    d = [(0, 0), (0, 1), (0, 2), (0, 3),
         (1, 0), (1, 1), (1, 2), (1, 3)]
    row = ord(move[0]) - ord('a')
    col = int(move[1:]) - 1
    flip, rot = d[direction]
    if flip:
        row = 14 - row
        col = 14 - col
    m, n = row - 7, col - 7
    for i in range(rot):
        m, n = -n, m
    row, col = m + 7, n + 7
    if row < 0 or row > 14 or col < 0 or col > 14:
        print('move is {}, direction {} {}, m n {} {}, row col {} {}'.format(
                move, flip, rot, m, n, row, col))
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

def error(s):
    print('error: {}'.format(s))

class ComplexBoard:
    def __init__(self):
        self.black_board = [[0 for i in range(15)] for j in range(15)]
        self.white_board = [[0 for i in range(15)] for j in range(15)]
        self.b_data = [[[0 for i in range(15)] for j in range(15)] for k in range(4)]
        self.w_data = [[[0 for i in range(15)] for j in range(15)] for k in range(4)]
        self.b_blocked = [[[0 for i in range(15)] for j in range(15)] for k in range(4)]
        self.w_blocked = [[[0 for i in range(15)] for j in range(15)] for k in range(4)]
        
    def move(self, pos, side):
        dir = [(1, 0), (1, 1), (0, 1), (-1, 1)]
        if side == 0:
            my_board = self.black_board
            op_board = self.white_board
            my_data = self.b_data
            op_data = self.w_data
            my_blocked = self.b_blocked
            op_blocked = self.w_blocked
        else:
            my_board = self.white_board
            op_board = self.black_board
            my_data = self.w_data
            op_data = self.b_data
            my_blocked = self.w_blocked
            op_blocked = self.b_blocked
        x, y = pos
        if my_board[x][y] != 0:
            error('already occupied')
        my_board[x][y] = 1

        # set all threats to 0 at this point
        for i in range(4):
            my_data[i][x][y] = 0
            my_blocked[i][x][y] = 0
            op_data[i][x][y] = 0
            op_blocked[i][x][y] = 0

        for i in range(4):
            dx, dy = dir[i]
            p_blocked = 0
            p = 1
            for j in range(1, 15):
                xi = x + j*dx
                yi = y + j*dy
                if (xi < 0 or xi > 14 or
                    yi < 0 or yi > 14 or
                    op_board[xi][yi] == 1):
                    p_blocked = 1
                    break
                elif my_board[xi][yi] == 1:
                    p += 1
                else:
                    break
            n_blocked = 0
            n = 1
            for j in range(1, 15):
                xi = x - j*dx
                yi = y - j*dy
                if (xi < 0 or xi > 14 or
                    yi < 0 or yi > 14 or
                    op_board[xi][yi] == 1):
                    n_blocked = 1
                    break
                elif my_board[xi][yi] == 1:
                    n += 1
                else:
                    break

            if not p_blocked:
                pxi = x + p*dx
                pyi = y + p*dy
                my_data[i][pxi][pyi] += n
                if n_blocked:
                    my_blocked[i][pxi][pyi] += 1
#                 if i == 2 and pxi == 8 and pyi == 11:
#                     import pdb
#                     pdb.set_trace()
            if not n_blocked:
                nxi = x - n*dx
                nyi = y - n*dy
                my_data[i][nxi][nyi] += p
                if p_blocked:
                    my_blocked[i][nxi][nyi] += 1
#                 if i == 2 and nxi == 8 and nyi == 11:
#                     import pdb
#                     pdb.set_trace()
            
            for j in range(1, 15):
                xi = x + j*dx
                yi = y + j*dy
                if (xi < 0 or xi > 14 or
                    yi < 0 or yi > 14):
                    break
                elif op_board[xi][yi] != 1:
                    if my_board[xi][yi] == 0:
                        op_blocked[i][xi][yi] += 1
                    break
                    
            
    def data(self, side):
        if side == 0:
            my_board = self.black_board
            op_board = self.white_board
            my_data = self.b_data
            op_data = self.w_data
            my_blocked = self.b_blocked
            op_blocked = self.w_blocked
        else:
            my_board = self.white_board
            op_board = self.black_board
            my_data = self.w_data
            op_data = self.b_data
            my_blocked = self.w_blocked
            op_blocked = self.b_blocked

        my_d = [[[[[0 for i in range(15)] for j in range(15)] 
               for k in range(3)] for m in range(4)] for s in range(4)]
        for direction in range(4):
            for i in range(15):
                for j in range(15):
                    n = my_data[direction][i][j]
                    b = my_blocked[direction][i][j]
                    if n > 0 and n < 4:
                        my_d[direction][n-1][b][i][j] = 1
                    if n >= 4:
                        my_d[direction][3][b][i][j] = 1
        
        op_d = [[[[[0 for i in range(15)] for j in range(15)] 
               for k in range(3)] for m in range(4)] for s in range(4)]
        for direction in range(4):
            for i in range(15):
                for j in range(15):
                    n = my_data[direction][i][j]
                    b = my_blocked[direction][i][j]
                    if n > 0 and n < 4:
                        op_d[direction][n-1][b][i][j] = 1
                    if n >= 4:
                        op_d[direction][3][b][i][j] = 1
        my_d = [x for z in my_d for y in z for x in y]
        op_d = [x for z in op_d for y in z for x in y]
        x = [my_board, op_board]+my_d+op_d
        return numpy.array(x)

def game2img(game, direction=0):
    '''
    input: a string that records the game, moves separated by spaces
    output: an array of matrices that represents the features, 
            and the array of actual next moves
    '''
    moves = [moveindex(move, direction) for move in game.split()]
    turn = 0
    board = ComplexBoard()
    b_data = []
    m_data = []
    
    for move in moves[:OPENING_N]:
        board.move(move, turn)
        turn = 1 - turn
    for move in moves[OPENING_N:]:
        b_data.append(board.data(turn))
        m_data.append(move)
        board.move(move, turn)
        turn = 1 - turn
    return numpy.array(b_data), m_data

def game_pos(game, winside, direction=0):
    '''
    input: a string that records the game, moves separated by spaces
    output: a random situation during the game
    '''
    moves = [moveindex(move, direction) for move in game.split()]
    board_black = numpy.zeros((15, 15), dtype='int32')
    board_white = numpy.zeros((15, 15), dtype='int32')
    board = [board_black, board_white]
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
    if winside == 'black':
        y = 1
    elif winside == 'white':
        y = -1
    else:
        y = 0
    if num % 2 == 1:
        board = board[1], board[0]
        y = -y
    return board, y

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
    N = 40000
    if DEBUG:
        root = root[:N]
    data = []

    n_validate = 100
    n_test = 100

    game_strings = []
    
    # train set with symmetry
    for i, game in enumerate(root[:-n_validate-n_test]):
        board_string = game.find('board').text
        if (not board_string or'--' in board_string):
            pass
        else:
            game_strings.append(board_string)
    shuffle(game_strings)
    v_data = []
    # test set with no symmetry
    for i, game in enumerate(root[-n_validate-n_test:-n_test]):
        board_string = game.find('board').text
        if not board_string or'--' in board_string:
            pass
        else:
            v_data.append(game2img(board_string))
        if i % 50 == 0:
            print('no{}'.format(i))

    t_data = []
    for i, game in enumerate(root[-n_test:]):
        board_string = game.find('board').text
        if not board_string or'--' in board_string:
            pass
        else:
            t_data.append(game2img(board_string))
        if i % 50 == 0:
            print('no{}'.format(i))

    print("total data: {}".format(len(data)))
    # print(data)
    MAX = 50
    v_data_x = [x.reshape(98*15*15) for d in v_data for x in d[0]]
    v_data_y = [y[0]*15 + y[1] for d in v_data for y in d[1]]
    t_data_x = [x.reshape(98*15*15) for d in t_data for x in d[0]]
    t_data_y = [y[0]*15 + y[1] for d in t_data for y in d[1]]
    with open('test_results', 'w') as f:
        print(t_data_y, file=f)
    test_xy = (t_data_x, t_data_y)
    with open('test_case.pkl', 'wb') as f:
        pickle.dump(test_xy, f)
    validate_x, validate_y = shared_dataset((v_data_x,
                                             v_data_y))
    test_x, test_y = shared_dataset((t_data_x,
                                     t_data_y))
    length = len(game_strings)//MAX
    for round in range(length):
        data = []
        for i in range(MAX):
            for d in range(8):
                data.append(game2img(game_strings[round*MAX+i], d))
        shuffle(data)
        data_x = [x.reshape(98*15*15) for d in data for x in d[0]]
        data_y = [y[0]*15 + y[1] for d in data for y in d[1]]
        print(len(data))
        train_x, train_y = shared_dataset((data_x, data_y))
        rval = [(train_x, train_y), (validate_x, validate_y), (test_x, test_y)]
        yield rval

def ol_win_data(filename):
    '''
    input: string
    output: a random position of the board
    '''
    tree = ET.parse(filename)
    root = tree.getroot()
    N = 50000
    n_validate = 2000
    n_test = 2000
    n_train = N - n_validate - n_test
    if DEBUG:
        root = root[:N]
    data = []
    for i, game in enumerate(root[:-n_validate-n_test]):
        if i % 50 == 0:
            print('no{}'.format(i))
        board_string = game.find('board').text
        if not board_string or'--' in board_string:
            continue
        winside = game.find('winner').text
        reason = game.find('winby').text
        if not reason in ['resign', 'five'] or not winside in ['black', 'white']:
            continue
        for direction in range(8):
            data.append(game_pos(board_string, winside, direction))
            data.append(game_pos(board_string, winside, direction))
            data.append(game_pos(board_string, winside, direction))
            data.append(game_pos(board_string, winside, direction))
            data.append(game_pos(board_string, winside, direction))
    print("total data: {}".format(len(data)))
    shuffle(data)
    data_x = [d[0].reshape(2*15*15) for d in data]
    data_y = [d[1] for d in data]

    v_data = []
    for i, game in enumerate(root[-n_validate-n_test:-n_test]):
        if i % 50 == 0:
            print('no{}'.format(i))
        board_string = game.find('board').text
        if not board_string or'--' in board_string:
            continue
        winside = game.find('winner').text
        reason = game.find('winby').text
        if not reason in ['resign', 'five'] or not winside in ['black', 'white']:
            continue
        v_data.append(game_pos(board_string, winside))
    v_data_x = [d[0].reshape(2*15*15) for d in v_data]
    v_data_y = [d[1] for d in v_data]

    t_data = []
    for i, game in enumerate(root[-n_test:]):
        if i % 50 == 0:
            print('no{}'.format(i))
        board_string = game.find('board').text
        if not board_string or'--' in board_string:
            continue
        winside = game.find('winner').text
        reason = game.find('winby').text
        if not reason in ['resign', 'five'] or not winside in ['black', 'white']:
            continue
        t_data.append(game_pos(board_string, winside))

    t_data_x = [d[0].reshape(2*15*15) for d in t_data]
    t_data_y = [d[1] for d in t_data]
    
    n_train = n_train * 8

    print('total train data: {}'.format(n_train))
    with open('win_test_results', 'w') as f:
        print(t_data_y, file=f)
    with open('win_test_case.pkl', 'wb') as f:
        pickle.dump(t_data, f)
    train_x, train_y = shared_dataset((data_x,
                                       data_y))
    validate_x, validate_y = shared_dataset((v_data_x,
                                             v_data_y))
    test_x, test_y = shared_dataset((t_data_x,
                                     t_data_y))
    rval = [(train_x, train_y), (validate_x, validate_y), (test_x, test_y)]
    return rval


if __name__ == '__main__':
#     numpy.set_printoptions(threshold=numpy.nan)
#     print(ol_win_data('../data/games.xml'))
#     ol_win_data('../data/games.xml')
#     a = numpy.zeros(15*15).reshape((15, 15))
#     a[[4, 5, 6]] = 1
#     b = []
#     for i in range(15):
#         for j in range(15):
#             b.append(1 if check_n(a, (i, j), 5) else 0)
#     x = numpy.array(b).reshape((15, 15))
#     print(x)
    ol_move_data('../data/games.xml')

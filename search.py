import pickle

import numpy
import theano
import theano.tensor as T

from policy import ConvNetwork

class Board:
    def __init__(self):
        self.board = [[0] * 15 for i in range(15)]
        self.move = 0
        self.movelist = []
        self.network = ConvNetwork()
        self.network.load(pickle.load(open('model_344999.mod', 'rb')))

    def print(self):
        print('  ', end='')
        for i in range(10):
            print(' {}  '.format(i), end='')
        for i in range(10, 15):
            print(' {} '.format(i), end='')
        print('')
        for i in range(15):
            if i < 10:
                print(' {}'.format(i), end='')
            else:
                print(i, end='')
            for j in range(15):
                if self.board[i][j] == 1:
                    print(' O ', end='')
                elif self.board[i][j] == -1:
                    print(' X ', end='')
                else:
                    print('   ', end='')
                if j != 14:
                    print('|', end='')
            print('')
            print('  ', end='')
            if i != 14:
                for j in range(15):
                    print('---', end='')
                    if j != 14:
                        print('*', end='')
                print('')
    
    def start(self):
        for i in range(15):
            for j in range(15):
                self.board[i][j] = 0
        self.move = 0
        self.movelist = []

    def moveto(self, x, y):
        if not self.board[x][y] == 0:
            print('error')
        else:
            if self.move % 2 == 0:
                turn = 1
            else:
                turn = -1
            self.board[x][y] = turn
            self.move += 1
            self.movelist.append((x, y))
    
    def undo(self):
        if self.move == 0:
            print('no move to undo')
        else:
            x, y = self.movelist[-1]
            self.board[x][y] = 0
            self.move -= 1

    def suggest(self, go=False):
        board_b = [[0] * 15 for i in range(15)]
        board_w = [[0] * 15 for i in range(15)]
        if self.move % 2 == 0:
            turn = 1
        else:
            turn = -1
        for i in range(15):
            for j in range(15):
                if self.board[i][j] == turn:
                    board_b[i][j] = 1
                elif self.board[i][j] == -turn:
                    board_w[i][j] = 1
        x = numpy.array([board_b, board_w])
        x = [x.reshape(2*15*15)] * 40
        shared_x = theano.shared(numpy.asarray(x, dtype=theano.config.floatX),
                                 borrow=True)
        result = self.network.predict(shared_x)[0].argmax()
        x, y = result//15, result%15
        if go:
            self.moveto(x, y)
        else:
            print('suggest: {} {}'.format(x, y))
            return x, y

if __name__ == '__main__':
    instr = ''
    board = Board()
    while True:
        instr = input('command: ')
        if instr == 'n':
            board.start()
        elif instr.startswith('m'):
            x, y = instr[1:].split()
            x = int(x)
            y = int(y)
            board.moveto(x, y)
        elif instr == 'u':
            board.undo()
        elif instr == 'g':
            board.suggest(go=True)
        elif instr == 'q':
            exit()
        board.print()
        board.suggest()
#     network = ConvNetwork()
#     network.load(pickle.load(open('model_344999.mod', 'rb')))
#     board_b[7][7] = 1
#     board_b[7][9] = 1
#     board_w[7][6] = 1
#     x = numpy.array([board_b, board_w])
#     x = [x.reshape(2*15*15)] * 40
#     shared_x = theano.shared(numpy.asarray(x, dtype=theano.config.floatX),
#                              borrow=True)
#     result = network.predict(shared_x)[0].argmax()
#     x, y = result//15, result%15
#     print(x, y)

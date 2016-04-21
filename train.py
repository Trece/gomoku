import xml.etree.ElementTree as ET
from pprint import pprint

import numpy
import theano
import theano.tensor as T

DEBUG = True

def moveindex(move):
    row = ord(move[0]) - ord('a')
    col = int(move[1:]) - 1
    return row, col

def ol_data(filename):
    '''
    input: string
    output: a list of tuples per yield, each tuple represents a move
    '''
    tree = ET.parse(filename)
    root = tree.getroot()
    if DEBUG:
        root = root[:10]
    for game in root:
        board_string = game.find('board').text
        yield [moveindex(move) for move in board_string.split()]



if __name__ == '__main__':
    for game in ol_data('games.xml'):
        pprint(moves)

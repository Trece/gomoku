import pickle

import numpy as np
from scipy.misc import imsave
from scipy.misc import imshow

class BoardImg:
    SCALE = 41
    white = np.array([255, 255, 255])
    black = np.array([0, 0, 0])
    green = np.array([0, 255, 0])
    blue = np.array([0, 0, 255])
    tan = np.array([210, 188, 180])
    def __init__(self):
        self.board = np.zeros((15, 15))
        self.height = 15 * self.SCALE
        self.width = 15 * self.SCALE
        self.base_image = np.zeros((self.height, self.width, 3), dtype='uint8')
        for i in range(self.base_image.shape[0]):
            for j in range(self.base_image.shape[1]):
                self.base_image[i, j] = self.tan
        for i in range(self.base_image.shape[0]):
            for j in range(self.base_image.shape[1]):
                if (i % self.SCALE == self.SCALE // 2 
                    or j % self.SCALE == self.SCALE //2):
                    self.base_image[i, j] = self.black
    
    def move(self, pos, color, p=1.0):
        x, y = pos
        img_x = self.SCALE * x + self.SCALE//2
        img_y = self.SCALE * y + self.SCALE//2
        r = self.SCALE//2
        for i in range(img_x - r, img_x + r):
            for j in range(img_y - r, img_y + r):
                if (i-img_x)*(i-img_x) + (j-img_y)*(j-img_y) < r * r:
                    self.base_image[i, j] += np.array(
                        p * (color - self.base_image[i, j]), dtype='uint8')

    def acmove(self, pos, color):
        x, y = pos
        img_x = self.SCALE * x + self.SCALE//2
        img_y = self.SCALE * y + self.SCALE//2
        r = self.SCALE//3 + 1
        for i in range(img_x - r, img_x + r):
            for j in range(img_y - r, img_y + r):
                if (i-img_x)*(i-img_x) + (j-img_y)*(j-img_y) < r * r:
                    self.base_image[i, j] = color
        

    def predictions(self, p):
        p = p.reshape((15, 15))
        for i in range(15):
            for j in range(15):
                self.move((i, j), self.blue, p[i, j])

    def save(self, name):
        imsave(name, self.base_image)

def read_data(filename):
    with open(filename) as f:
        s = f.read()
        s = s.replace('[', '').replace(']', '')
        return np.array(s.split(), dtype='float32')

def board_data(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
        return np.array(data[0]), np.array(data[1])

if __name__ == '__main__':
    all_p = read_data('test_4999.res')
    data_x, data_y = board_data('test_case.pkl')
    for n in range(40):
        board_img = BoardImg()
        x = data_x[n]
        y = data_y[n]
        x = x.reshape((2, 15, 15))
        print(x.shape)
        for i in range(15):
            for j in range(15):
                if x[0, i, j] == 1:
                    board_img.acmove((i, j), BoardImg.black)
        for i in range(15):
            for j in range(15):
                if x[1, i, j] == 1:
                    board_img.acmove((i, j), BoardImg.white)
        board_img.acmove((y//15, y%15), BoardImg.green)
        p = all_p[225*n:225*(n+1)]
        p *= 10
        board_img.predictions(p)
        board_img.save('boardbeg_{}.png'.format(n))
    

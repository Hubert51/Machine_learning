'''
LFD
Problem 3.1(b)
'''

import numpy as np
import matplotlib.pyplot as plt
import math
import random
import sys
from numpy.linalg import inv

def create_a_random_line():
    a = random.randint(-5,5)
    b = random.randint(-5,5)
    c = random.randint(-5,5)
    # x = np.arange(-5,5, 0.1)
    # y = 1.0*a * x/b + 1.0/b
    return a,b,c

def check_data(a, b,c, a_g, b_g,c_g,  x, y):
    result = True
    a1 = a * x + b * y + c
    b1 = a_g * x + b_g * y + c_g
    if (a * x + b * y + c > 0 and a_g * x + b_g * y + c_g <= 0 ):
        a_g += x
        b_g += y
        c_g += 30
        result = False
    elif (a * x + b * y + c <= 0 and a_g * x + b_g * y + c_g >= 0 ):
        a_g -= x
        b_g -= y
        c_g -= 30
        result = False

    return result, a_g,b_g,c_g


def create_data_set(thk,sep,rad,number):
    # 1 as positive data
    # zero as negative data
    classify = np.random.randint(0,2,number)
    pos_num = len(np.where(classify[:]==1)[0])
    neg_num = number - pos_num

    x_pos = np.random.uniform(0,2*(thk+rad), pos_num)
    x_neg = np.random.uniform(rad+0.5*thk, 3*rad+2.5*thk, neg_num)

    # create positive y
    y_pos_low_bound = np.sqrt(rad**2 - np.square(thk+rad-x_pos))
    y_pos_high_bound = np.sqrt((rad+thk)**2 - np.square(thk+rad-x_pos))
    y_pos = np.zeros_like(x_pos,dtype=np.float32)
    y_neg = np.zeros_like(x_neg,dtype=np.float32)

    index = np.where( (thk < x_pos) & (x_pos < thk+2*rad) )
    index = np.array(index[0])
    index_complment = np.arange(0,pos_num,1)
    index_complment = np.setdiff1d(index_complment,index)
    offset = rad+sep+thk
    y_pos[index] = np.random.uniform(y_pos_low_bound[index],y_pos_high_bound[index], len(index)) + offset
    y_pos[index_complment] = np.random.uniform(0,y_pos_high_bound[index_complment], len(index_complment)) + offset

    # create negative y
    y_neg_low_bound = np.sqrt(rad**2 - np.square(1.5*thk+2*rad-x_neg))
    y_neg_high_bound = np.sqrt((rad+thk)**2 - np.square(1.5*thk+2*rad-x_neg))
    index = np.where( (1.5*thk+rad < x_neg) & (x_neg < 1.5*thk+3*rad) )
    index = np.array(index[0])
    index_complment = np.setdiff1d(np.arange(0, neg_num, 1),index)
    offset = 0
    y_neg[index] = (thk+rad) - np.random.uniform(y_neg_low_bound[index], y_neg_high_bound[index], len(index)) + offset
    y_neg[index_complment] = np.random.uniform((thk+rad), (thk+rad)-y_neg_high_bound[index_complment], len(index_complment)) + offset

    # plt.scatter(x_pos, y_pos, color="red")
    # plt.scatter(x_neg,y_neg, color="blue")
    # plt.show()
    return x_pos, x_neg, y_pos, y_neg


if __name__ == '__main__':
    thk = 5
    sep = 5
    rad = 10
    number = 2000
    random.seed(5)
    np.random.seed(22)

    # create number of data points
    x_pos, x_neg, y_pos, y_neg = create_data_set(thk,sep,rad,number)
    # use random function to get two random integer
    a_f = 0
    b_f = 1
    c_f = -(thk+rad+0.5*sep)
    x = np.arange(0,3*rad+3*thk, 0.1)
    # a_f,b_f,c_f = create_a_random_line()
    y_f = - (a_f*x + c_f) / b_f
    xdata = np.concatenate( (x_pos,x_neg) ).reshape((number,1))
    ydata = np.concatenate( (y_pos,y_neg) ).reshape((number,1))
    bias = np.ones_like(xdata).reshape((number,1))

    data_matrix = np.concatenate((bias, xdata,ydata),axis=1)
    target_vector = np.ones_like(bias)
    target_vector[0:len(x_pos)] = -1

    pseudo_inverse = inv(data_matrix.T.dot(data_matrix)).dot(data_matrix.T)
    # first weight is bias
    # second weight is for x
    # third weight is for y
    weight = pseudo_inverse.dot(target_vector)
    a_g = weight[1][0]
    b_g = weight[2][0]
    c_g = weight[0][0]

    y_g = - (a_g * x + c_g) / b_g
    hypothesis_line, = plt.plot(x, y_g, "r--", label='hypothesis g')
    plt.scatter(x_neg, y_neg, color="Blue")
    plt.scatter(x_pos, y_pos, color="Red")
    plt.legend([hypothesis_line],["hypothesis g: {:.2f}x+{:.2f}y+({:.2f})".format(a_g,b_g,c_g)])
    plt.title("PLA method to show data and final hypothesis")
    plt.xlabel("x"); plt.ylabel("y")
    plt.show()

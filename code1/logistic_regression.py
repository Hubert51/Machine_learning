# first import package
import numpy as np
import matplotlib.pyplot as plt
import cv2
import random
from numpy.linalg import inv
import matplotlib.cm as cm
from math import e
from sklearn.linear_model import LogisticRegression
import sys
from numpy.linalg import norm




def sigmoid(scores):
    return 1 / (1 + np.exp(-scores))


def log_likelihood(features, target, weights):
    scores = np.dot(features, weights)
    ll = np.sum( target*scores - np.log(1 + np.exp(scores)) )
    return ll


def logistic_regression(features, target, num_steps, learning_rate, add_intercept=False):
    if add_intercept:
        intercept = np.ones((features.shape[0], 1))
        features = np.hstack((intercept, features))

    weights = np.zeros(features.shape[1])

    for step in range(num_steps):
        scores = np.dot(features, weights)
        predictions = sigmoid(scores)

        # Update weights with gradient
        output_error_signal = target - predictions
        gradient = np.dot(features.T, output_error_signal)
        weights += learning_rate * gradient

        # Print log-likelihood every so often
        if step % 10000 == 0:
            pass
            # print(log_likelihood(features, target, weights))
            # print(gradient)
    return weights



def normalize(data_set):
    # transfer feature one by one
    data_set = data_set.astype(np.float32)
    for i in range(1, len(data_set[0])):
        max1 = np.max(data_set[:, i])
        min1 = np.min(data_set[:, i])
        diff = np.max(data_set[:, i]) + np.min(data_set[:, i])
        data_set[:, i] = 1.0 * (data_set[:, i] - min1 - (max1 - min1) / 2) / ((max1 - min1) / 2)
    return data_set


def data_process(files):
    digit1, not_digit1 = [], []
    for file in files:
        raw_data = np.loadtxt(file)
        data = raw_data[:, 1:]
        for index in range(len(data)):
            number = data[index].reshape((16, 16))
            # feature 1. whether vertical symmetric
            number_flip = cv2.flip(number, 0)
            # more count means more unsymmetrical
            count = len(np.where(number != number_flip)[0])
            # this is feature for intensity range
            intensity = len(np.where(number > -1.0)[0])
            ## value can be from 0 to 9
            digit1.append((int(raw_data[index, 0]), count, intensity))
    digit1 = np.array(digit1)
    return digit1


def split_data(data_set):
    random.seed(20)
    value = [[0, 475, ]]
    pos = data_set[np.where(data_set[:, 0] == 0)]
    neg = data_set[np.where(data_set[:, 0] == 4)]
    pos[:, 0] = 1
    neg[:, 0] = 0

    pos_train_index = np.array(random.sample(range(len(pos)), 475))
    neg_train_index = np.array(random.sample(range(len(neg)), 25))
    pos_test_index = np.delete(np.arange(len(pos)), pos_train_index)
    neg_test_index = np.delete(np.arange(len(neg)), neg_train_index)

    pos_train_data = pos[pos_train_index]
    pos_test_data = pos[pos_test_index]
    neg_train_data = neg[neg_train_index]
    neg_test_data = neg[neg_test_index]
    return pos_train_data, pos_test_data, neg_train_data, neg_test_data


# def log_likelihood(features, target, weights):
#     scores = np.dot(features, weights)
#     ll = np.sum(target * scores - np.log(1 + np.exp(scores)))
#     return ll

def my_calculate_weight(x_train, y_train):
    N = len(y_train)
    w = np.zeros(3)
    const = 10
    old_w = 0
    y_train = y_train.reshape(500, 1)

    for itr in range(10000):
        yx = y_train * (x_train)
        w = w.reshape(3,1)
        denominator = (1 + e ** (y_train * x_train.dot(w) ))
        g = np.sum(yx/denominator,0)/N
        g = g.reshape(3,1)

        w += const * g
        if itr % 100 == 0:
            # I set a stop point when the difference of old and new weight
            # is too small
            if abs(norm(w) - norm(old_w)) < 1e-6:
                break
            old_w = np.copy(w)
    return w


def my_logistic_regression(train_data_set, test_data_set):
    # training data
    x_train = np.copy(train_data_set)
    x_train[:, 0] = 1
    y_train = np.array(train_data_set[:, 0])
    y_train[np.where(y_train == 0)] = -1
    # test data
    x_test = np.copy(test_data_set[:, :])
    y_test = np.array(test_data_set[:, 0])
    w = my_calculate_weight(x_train, y_train)
    x_test[:, 0] = 1
    scores = np.dot(x_test, w)
    predictions = np.round(sigmoid(scores))
    result = np.concatenate((predictions.reshape(len(predictions), 1), y_test.reshape(len(y_test), 1)), 1)
    print("Accuracy of our logistic {}".format(len(np.where(result[:, 0] == result[:, 1])[0]) / len(result)))


def SK_Learn_logistic_regression(train_data_set, test_data_set):
    logisticRegr = LogisticRegression(fit_intercept=True, C=1e15)
    logisticRegr.fit(train_data_set[:, 1:], train_data_set[:, 0])
    score = logisticRegr.score(test_data_set[:, 1:], test_data_set[:, 0])
    print("Accuracy from SK-Learn {}".format(score))


def other_my_logistic_regression(train_data_set, test_data_set):

    weights = logistic_regression(train_data_set[:,1:], train_data_set[:,0],
                                  num_steps=300000, learning_rate=5e-5, add_intercept=True)
    x_test = np.copy(test_data_set[:, :])
    y_test = np.array(test_data_set[:, 0])
    x_test[:, 0] = 1
    scores = np.dot(x_test, weights)
    predictions = np.round(sigmoid(scores))
    result = np.concatenate((predictions.reshape(len(predictions), 1), y_test.reshape(len(y_test), 1)), 1)
    print("Accuracy of other logistic {}".format(len(np.where(result[:, 0] == result[:, 1])[0]) / len(result)))


if __name__ == '__main__':
    # read data from file
    train_file = "ZipDigits.train.txt"
    test_file = "ZipDigits.test.txt"
    order = 8
    data_set = data_process([train_file, test_file])
    # normalize feature
    data_set = normalize(data_set)
    # draw the number with different color to check the distribution.
    # colors = cm.rainbow(np.linspace(0, 1, 10))
    # for i in range(10):
    #     if i != 0 and i != 4:
    #         continue
    #     index = np.where(data_set[:, 0] == i)
    #     print("The size of digit {} is {}".format(i, len(index[0])))
    #     plt.scatter(data_set[index, 1], data_set[index, 2], color=colors[i], label="digit " + str(i))
    # plt.legend(fontsize='small')
    # plt.show()

    # number of total training data
    N = 500
    data_set = data_set[np.where((data_set[:, 0] == 0) | (data_set[:, 0] == 4))]
    p_train, p_test, n_train, n_test = split_data(data_set)

    # paras = [[1, p_train, "blue", "0"], [0, n_train, "red", "4"]]
    # for para in paras:
    #     train = para[1]
    #     index = np.where(train[:, 0] == para[0])
    #     plt.scatter(train[index, 1], train[index, 2], color=para[2], label="digit " + str(para[3]))
    # plt.title("Training Data for Digit 0 Versus Digit 4")
    # plt.xlabel("symmetric")
    # plt.ylabel("Intensity")
    # plt.legend(fontsize='small')
    # plt.show()

    train_data_set = np.concatenate((p_train, n_train), 0)
    test_data_set = np.concatenate((p_test, n_test), 0)

    my_logistic_regression(train_data_set, test_data_set)
    SK_Learn_logistic_regression(train_data_set, test_data_set)
    other_my_logistic_regression(train_data_set, test_data_set)

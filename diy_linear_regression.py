'''
diy linear regression learning algorithm

specifically: use height and gender as independent variables
to estimate the weight of a person.
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# for comparison
from sklearn.linear_model import LinearRegression

# easy progress bar
import tqdm


def mse(y, y_pred):
    '''mean square error between labels, predictions'''
    # print((y - y_pred).T)
    e = np.dot((y - y_pred).T, (y - y_pred))
    return e / len(y)


class DIYLinearRegression():
    '''
    implements linear regression;
    a rough analog to sklearn.linear.LinearRegression.
    additionally stores error and accuracy histories as lists.

    args:
        X, y: ndarrays with matching numbers of samples

        metric: a function that maps (y, preds) to loss
    '''
    def __init__(self, X, y, metric=mse, verbose=1):
        print('logistic regression on with {} as metric'.format(metric))
        self.X = np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)
        self.y = y.copy()
        self.n = X.shape[0]   # number of samples
        self.step = .001       # distance for gradient measurement
        self.rate = .1         # how many times gradient to step
        self.metric = metric  # accuracy or other metric

        self.w = np.random.randn(X.shape[1] + 1)  # bias added to w
        print('initialized weights to:', self.w)

        self.err_history = []
        self.v = verbose
        if self.v:
            print('X shape', self.X.shape)

        for i in range(1):
            print('Random sample sanity check:',
                  self.X[np.random.randint(0, self.n)])

    def dumb_gradient(self):
        '''
        finds gradient of metric with respect to current weights:
        d(metric) / dw

        'Dumb' because it is not determined analytically
        but instead measured empirically across all samples.

        returns:
            gradient, error(metric), accuracy
        '''
        error = self.metric(self.y, self.predict(self.X))
        grad = []
        for i in range(len(self.w)):
            self.w[i] += self.step
            # print(self.w)
            new_error = self.metric(self.y, self.predict(self.X))
            self.w[i] -= self.step
            grad.append((error - new_error) / self.step)

        # grad is the change in error when w[i] is increased by self.step
        # grad = dE/dw for w in weights
        # if grad[i] is positive, increasing w[i] increases error in predictions
        return np.array(grad), error

    def _fit(self):
        '''train one epoch and print'''
        grad, error = self.dumb_gradient()
        if self.v:
            print('weights : {:f} {:f} {:f}'.format(*self.w))
            print('gradient: {:.1f} {:.1f} {:.1f}'.format(*grad))
            print('error:    {}'.format(error))

        self.w = (grad * self.rate / error) + self.w
        self.err_history.append(error)

    def fit(self, n):
        '''train n epochs'''
        if self.v:
            for i in range(n):
                self._fit()
        else:
            for i in tqdm.tqdm(range(n)):
                self._fit()

    def predict(self, x):
        '''predict data'''
        return np.dot(self.w, x.T)

    def display(self):
        # make masks
        fig, axes = plt.subplots(nrows=1, ncols=1)
        plt.title = 'gradient metric: {}'.format(self.metric)

        # subplot 2: matplotlib
        plt.ylabel('score')
        plt.xlabel('epoch')
        plt.plot(self.err_history, 'rx', label='loss')
        plt.legend()

        plt.show()


def main():
    # import data
    df = pd.read_csv('data/heights_weights_genders.csv')
    df.Gender = (df.Gender == 'Male').astype(int)

    X = df[['Height', 'Gender']].values
    y = df.Weight.values

    # baseline
    print()
    sklr = LinearRegression()
    sklr.fit(X, y)
    print('Baseline SKlearn score:', sklr.score(X, y))
    print()

    # diy
    lr = DIYLinearRegression(X, y, metric=mse, verbose=0)
    lr.fit(3000)
    lr.display()

    return lr


if __name__ == '__main__':
    model = main()

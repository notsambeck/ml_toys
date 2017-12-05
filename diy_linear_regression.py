'''
diy linear regression learning algorithm

use case: use height and gender as independent variables
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
    implements linear regression on X, y;
    a rough analog to sklearn LinearRegression.

    dataset is standardized and stored with bias column added;
    x_offset and x_scale allow new samples to be predicted.

    additionally stores error  history as a list.

    args:
        X, y : ndarrays of data, labels
        metric : a function that maps (y, preds) to loss
    '''
    def __init__(self, X, y, metric=mse, standardize=True, verbose=1):
        self.step = .001       # distance for gradient measurement
        self.rate = 100         # how many times gradient to step
        self.metric = metric   # accuracy or other metric
        self.err_history = []
        self.v = verbose

        self.X = np.concatenate([np.ones((X.shape[0], 1)), X.copy()], axis=1)
        self.y = y.copy()
        self.n = self.X.shape[0]    # number of samples
        self.w = np.random.randn(X.shape[1] + 1)

        self.stdz = standardize
        if self.stdz:
            self.standardize()

        if self.v:
            print('logistic regression on with {} as metric'.format(metric))
            print('X.shape', self.X.shape)
            print('initialized weights to:', self.w)
            for i in range(self.v):
                print('Random sample sanity check:',
                      self.X[np.random.randint(0, self.n)])

    def standardize(self, data=None):
        self.x_offset = np.mean(self.X, axis=0)
        self.x_offset[0] = 0   # keep bias
        self.X = self.X - self.x_offset
        if self.v:
            print('x_offset =', self.x_offset)

        self.x_scale = np.concatenate([np.ones(1), np.std(self.X[:, 1:], axis=0)],
                                      axis=0)
        self.X = self.X / self.x_scale
        if self.v:
            print('x_scale =', self.x_scale)

        return self.x_offset, self.x_scale

    def dumb_gradient(self):
        '''
        finds gradient of metric with respect to current weights:
        d(metric) / dw

        'Dumb' because it is not determined analytically
        but instead measured empirically across all samples.

        returns:
            gradient, error(metric), accuracy
        '''
        error = self.metric(self.y, np.dot(self.w, self.X.T))
        grad = []
        for i in range(len(self.w)):
            self.w[i] += self.step
            # print(self.w)
            new_error = self.metric(self.y, np.dot(self.w, self.X.T))
            self.w[i] -= self.step
            grad.append((error - new_error) / self.step)

        # grad is the change in error when w[i] is increased by self.step
        # i.e. grad = dE/dw for w in weights
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
        if self.err_history and error > self.err_history[-1]:
            self.rate /= 2
            if self.v:
                print('halve learning rate')
        self.err_history.append(error)

    def fit(self, n=150):
        '''train n epochs; show either verbose output or progress bar'''
        if self.v:
            for i in range(n):
                self._fit()
        else:
            for i in tqdm.tqdm(range(n)):
                self._fit()

    def predict(self, x):
        '''standardize, then predict for new data'''
        if type(x) is list:
            x = np.array(x)
        # add bias if missing
        if x.shape[1] == self.w.shape[0] - 1:
            x = np.concatenate([np.ones((x.shape[0], 1)), x], axis=1)
        elif x.shape[1] == self.w.shape[0]:
            pass
        else:
            print('x.shape=', x.shape)
            raise ValueError('new data must match shape of weights')
        x = (x - self.x_offset) / self.x_scale
        return self._predict(x)

    def _predict(self, x):
        '''for standardized data'''
        return np.dot(self.w, x.T)

    def score(self):
        # coefficient of determination to match sklearn lr.score
        u = self.metric(self.y, self._predict(self.X))
        v = self.metric(self.y, np.ones(self.n) * self.y.mean())
        return 1 - u/v

    def display(self):
        # make masks
        fig, axes = plt.subplots(nrows=1, ncols=2)
        axes[0].set_title('heights/weights')
        axes[1].set_title('metric: {}'.format(self.metric))

        male_x = np.compress(self.X[:, 2] > 0, self.X, axis=0)
        male_y = np.compress(self.X[:, 2] > 0, self.y, axis=0)

        female_x = np.compress(self.X[:, 2] < 0, self.X, axis=0)
        female_y = np.compress(self.X[:, 2] < 0, self.y, axis=0)

        axes[0].scatter(male_x[:, 1], male_y, alpha=.03)
        axes[0].scatter(female_x[:, 1], female_y, alpha=.03, color='r')

        # show linear predictions: male
        line = np.stack([np.ones(100), np.linspace(-3, 3, 100), np.ones(100)]).T
        axes[0].plot(line[:, 1], self._predict(line), color='c', label='preds: male')

        # female
        line = np.stack([np.ones(100), np.linspace(-3, 3, 100), np.ones(100) * -1]).T
        axes[0].plot(line[:, 1], self._predict(line), color='pink', label='preds: female')

        plt.xlabel('height')
        plt.ylabel('weight')
        axes[0].legend()

        # subplot 2: matplotlib
        plt.ylabel('score')
        plt.xlabel('epoch')
        axes[1].plot(self.err_history, 'rx', label='loss')
        axes[1].legend()

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
    lr.fit()
    print('diy linear regression score:', lr.score())
    lr.display()

    return lr


if __name__ == '__main__':
    model = main()

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


def rmse(y, y_pred):
    '''rms error between labels, predictions'''
    return np.dot((y - y_pred), (y - y_pred).T) // len(y)


class DIYLinearRegression():
    '''
    implements linear regression;
    a rough analog to sklearn.linear.LinearRegression.
    additionally stores error and accuracy histories as lists.

    args:
        X, y: ndarrays with matching numbers of samples

        metric: a function that maps (y, preds) to loss
    '''
    def __init__(self, X, y, metric=rmse):
        print('logistic regression on with {} as metric'.format(metric))
        self.X = X.copy()
        self.y = y.copy()
        self.n = X.shape[0]
        self.step = .001      # distance for gradient measurement
        self.rate = 50        # how many times gradient to step
        self.metric = metric  # accuracy or other metric

        self.w = np.random.randn(X.shape[1] + 1)
        print('initialized weights to:', self.w)

        self.err_history = []
        self.acc_history = []

        self.means = []     # save statistics about columns of X
        self.stds = []      # in order to standardize additional data
        self.standardize()

        for i in range(1):
            print('Random sample sanity check:',
                  self.X[np.random.randint(0, self.n)])

    def dumb_gradient(self, metric=rmse):
        '''
        finds gradient of metric with respect to current weights:
        d(metric) / dw

        'Dumb' because it is not determined analytically
        but instead measured empirically across all samples.

        returns:
            gradient, error(metric), accuracy
        '''
        error = metric(self.y, self.predict(self.X))
        # print('error @ epoch start: {}; accuracy: {}'.format(error, acc))
        grad = []
        # print('input weights: {}: {}'.format(type(w), w))
        for i in range(len(self.w)):
            new_w = self.w.copy()
            new_w[i] += self.step
            # print(new_w)
            new_error = metric(self.y, self.predict(self.X))
            grad.append((error - new_error) / self.step)

        # grad is the change in error when w[i] is increased by .01
        # grad = dE/dw for w in weights
        # if grad[i] is positive,
        # print('unnormalized grad:{:.5}{:.5}{:.5}'.format(*[g for g in grad]))

        return np.array(grad), error

    def standardize(self):
        # standardize data
        print()
        print('standardizing data:')
        for i in range(self.X.shape[1]):
            # save mean, std so new values can be converted/standardize values
            self.means.append(self.X[:, i].mean())
            print('mean of X[{}]: {}'.format(i, self.means[i]))
            self.X[:, i] = np.subtract(self.X[:, i], self.means[i])
            self.stds.append(self.X[:, i].std())
            print('std. dev. of X[{}]: {}'.format(i, self.stds[i]))
            self.X[:, i] = np.divide(self.X[:, i], self.stds[i])     # / std
        print()

    def _train(self, v=1):
        '''train one epoch and print'''
        grad, error, acc = self.dumb_gradient(self.w,
                                              self.X,
                                              self.y,
                                              self.step,
                                              self.metric)
        if v:
            print('weights are: {:f} {:f} {:f}'.format(*self.w))
            print('gradient is: {:f} {:f} {:f}'.format(*grad))
            print('error: {:.5} accuracy: {:.2}'.format(error, acc))

        self.w = np.add(np.multiply(grad, self.rate), self.w)

        self.err_history.append(error)
        self.acc_history.append(acc)

    def train(self, n):
        '''train n epochs'''
        for i in tqdm.tqdm(range(n)):
            self._train(v=0)

    def predict_new_sample(self, x):
        '''predict new array of [heights, weights]'''
        x = np.subtract(x, self.means)
        x = np.divide(x, self.stds)
        return self.predict(x)

    def predict(self, x):
        '''predict already normalized data'''
        return np.dot(np.concatenate([x, np.ones(x.shape[0], 1)]), self.w)

    def display(self, df):
        # make masks
        df.Gender = (df.Gender == 'Male').astype(int)
        pred_weight = self.predict_new_sample(df[['Height', 'Gender']])

        y = df.Weight.values

        fig, axes = plt.subplots(nrows=1, ncols=2)
        plt.title = 'gradient metric: {}'.format(self.metric)

        # subplot 1: pandas
        plt.ylabel('weight')
        plt.xlabel('height')
        df[male].plot.scatter(ax=axes[0], x='Height', y='Weight', s=5,
                              alpha=.5, c='blue', label='male')
        df[~male].plot.scatter(ax=axes[0], x='Height', y='Weight', s=5,
                               alpha=.5, c='r', label='female')

        df[pred_male].plot.scatter(ax=axes[0], x='Height', y='Weight', s=.5,
                                   c='k', label='male_preds')
        df[~pred_male].plot.scatter(ax=axes[0], x='Height', y='Weight', s=.5,
                                    c='white', label='female_preds')

        # subplot 2: matplotlib
        plt.ylabel('score')
        plt.xlabel('epoch')
        plt.plot(self.err_history, 'rx', label='loss')
        plt.plot(self.acc_history, 'go', label='accuracy')
        plt.legend()

        plt.show()


def main():
    # import data
    df = pd.read_csv('data/heights_weights_genders.csv')
    df.Gender = (df.Gender == 'Male').astype(int)

    X = df[['Height', 'Weight']].values
    y = df.Gender.values

    # baseline
    print()
    sklr = LinearRegression()
    sklr.fit(X, y)
    print('Baseline SKlearn score:', sklr.score(X, y))
    print()

    # diy
    lr = DIYLinearRegression(X, y, metric=rmse)
    lr.train(150)
    lr.display(df)


if __name__ == '__main__':
    main()

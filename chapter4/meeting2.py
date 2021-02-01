import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression

# Logistic Regression on Data


def get_heart_datset():
    return pd.read_csv("../datasets/SAheart/SAheart.data", index_col=0)


def norm(x):
    x = (x - x.mean()) / x.std()
    return x


def get_sigma(y, y_hat, p):
    y_delta = y - y_hat
    y_delta_sq = y_delta.T @ y_delta
    sigma = np.sqrt(y_delta_sq / (len(y) - p - 1))
    return sigma


def get_std_errors(x, y, y_hat):
    x_cpy = x.copy()
    x_cpy["intercept"] = 1
    df = x_cpy.transpose().dot(x_cpy)
    inv = np.linalg.inv(df.values) * get_sigma(y, y_hat, x.shape[1])
    return np.sqrt(np.diag(inv))


def logistic_regression(x_train, y_train):
    regr = LogisticRegression(solver='newton-cg', penalty='none')
    regr.fit(x_train, y_train)
    return regr


if __name__ == '__main__':
    sa_heart = get_heart_datset()
    features = sa_heart.columns
    sa_heart["famhist"] = sa_heart["famhist"].astype('category').cat.codes
    selected_features = [features[i] for i in [0, 1, 2, 4, 6, 7, 8]]
    print(selected_features)
    x = sa_heart[selected_features]
    y = sa_heart["chd"]

    regr = logistic_regression(x, y)
    y_hat = regr.predict(x)
    std_errors = get_std_errors(x, y, y_hat)

    print("Intercept: %2.3f" % regr.intercept_)
    for i in range(len(selected_features)):
        print(selected_features[i]+": %2.3f , %2.5f"%(regr.coef_[0][i], std_errors[i] ))

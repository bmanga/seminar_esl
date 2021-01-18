import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd


def get_vowel_datsets():
    train = np.loadtxt("datasets/vowel/vowel.train.txt", delimiter=',', skiprows=1, usecols=(i for i in range(1, 12)))
    test = np.loadtxt("datasets/vowel/vowel.test.txt", delimiter=',', skiprows=1, usecols=(i for i in range(1, 12)))
    return train, test


def norm(x):
    x = (x - x.mean()) / x.std()
    return x


# Function for calculating error rate:
def get_error_rate(y_calc, y):
    """Gets the prediction error rate for linear regression

    Parameters
    ----------
    y_calc : ndarray
        The output of reg.predict
    y : DataFrame
        The correct output for comparation

    Returns
    -------
    float
        the prediction error rate
    """

    y_dummy_calc = pd.get_dummies(y_calc.argmax(axis=1))
    y_dummy_calc.columns = y.columns.values
    y_dummy_calc.index = y.index

    return np.mean(np.mean((y_dummy_calc != y) * 11 / 2))


def calc_linear_regression(x_train, y_train, x_test, y_test):
    # 1) LINEAR REGRESSION
    # Convert to dummy variables for better applicate linear regression
    y_dummy = pd.get_dummies(y_train)
    y_test_dummy = pd.get_dummies(y_test)
    # Fit the model
    reg = LinearRegression().fit(x_train, y_dummy)

    # Get the error for training and test
    print("Linear regression:")
    y_test_calc = reg.predict(x_test)
    y_calc = reg.predict(x_train)
    print("\tThe error rate on train is %2.2f %%" % get_error_rate(y_calc, y_dummy))
    print("\tThe error rate on test is %2.2f %%" % get_error_rate(y_test_calc, y_test_dummy))


def calc_lda(x_train, y_train, x_test, y_test):
    # 2) LDA
    # Fit the model (no need for dummy variables)
    model = LinearDiscriminantAnalysis(solver='eigen', shrinkage=None, priors=None,
                                       n_components=None, store_covariance=False, tol=0.0001)
    reg = model.fit(x_train, y_train)
    print("Linear discriminant analysis (LDA):")
    print("\tThe error rate on train is %2.5f %%" % (1 - reg.score(x_train, y_train)))
    print("\tThe error rate on test is %2.5f %%" % (1 - reg.score(x_test, y_test)))


def calc_qda(x_train, y_train, x_test, y_test):
    # 3) QDA
    # Fit the model (no need for dummy variables)
    model = QuadraticDiscriminantAnalysis()
    reg = model.fit(x_train, y_train)
    print("Quadratic discriminant analysis (QDA):")
    print("\tThe error rate on train is %2.5f %%" % (1 - reg.score(x_train, y_train)))
    print("\tThe error rate on test is %2.5f %%" % (1 - reg.score(x_test, y_test)))


def calc_logistic(x_train, y_train, x_test, y_test):
    # 3) QDA
    # Fit the model (no need for dummy variables)
    model = LogisticRegression(solver='newton-cg', penalty='none')
    reg = model.fit(x_train, y_train)
    print("Logistic regression:")
    print("\tThe error rate on train is %2.2f %%" % (1 - reg.score(x_train, y_train)))
    print("\tThe error rate on test is %2.2f %%" % (1 - reg.score(x_test, y_test)))


def discriminant_formula(x, sigma_inv, det, mean, pi):
    delta = float(-0.5 * np.log(det) - 0.5 * (x-mean) @ sigma_inv @ (x-mean) + np.log(pi))
    return delta


def get_graph_error(x, y, sigma_k_inv, det_k, reg):
    out = np.empty([x.shape[0], 11])
    for j in range(x.shape[0]):  # for each x
        out_row = np.empty(11)
        for i in range(11):  # for each possible output
            out_row[i] = discriminant_formula(x[j], sigma_k_inv[i], det_k[i], reg.means_[i], reg.priors_[i])
        out[j] = out_row
    y_dummy = pd.get_dummies(y)
    er = get_error_rate(out, y_dummy)
    return er


def get_errors_with_alpha(x_train, y_train, x_test, y_test, alpha):
    # Fit the model (no need for dummy variables)
    model = QuadraticDiscriminantAnalysis(store_covariance=True)
    reg = model.fit(x_train, y_train)
    # Equation 4.12
    det_k = []
    sigma_k_inv = []
    sigma = 0
    for sigma_i in reg.covariance_:
        sigma += sigma_i
    sigma = sigma/11

    for sigma_i in reg.covariance_:
        sigma_i = alpha * np.asmatrix(sigma_i) + (1 - alpha)*sigma
        det_i = np.linalg.det(sigma_i)
        det_k.append(det_i)
        sigma_i_inv = np.linalg.inv(sigma_i)
        sigma_k_inv.append(sigma_i_inv)

    train_error = get_graph_error(x_train, y_train, sigma_k_inv, det_k, reg)
    test_error = get_graph_error(x_test, y_test, sigma_k_inv, det_k, reg)
    return train_error, test_error


def create_graph(x_train, y_train, x_test, y_test):
    train_errors = []
    test_errors = []
    alpha_range = np.linspace(0, 1, 50)
    for alpha in alpha_range:
        tre, tee = get_errors_with_alpha(x_train, y_train, x_test, y_test, alpha)
        train_errors.append(tre)
        test_errors.append(tee)

    plt.scatter(alpha_range, train_errors, s=4, c='blue')
    plt.scatter(alpha_range, test_errors, s=4, c='orange')
    plt.show()


def create_seaborn_graph(x_train, y_train, x_test, y_test):
    train_errors = []
    test_errors = []
    alpha_range = np.linspace(0, 1, 50)
    for alpha in alpha_range:
        tre, tee = get_errors_with_alpha(x_train, y_train, x_test, y_test, alpha)
        train_errors.append(tre)
        test_errors.append(tee)

    # convert data into dataframe
    errors_df = pd.DataFrame([alpha_range, train_errors, test_errors]).transpose()
    # preprocessing data ( i don't know exactly how it works)
    errors_df = errors_df.rename(columns={0: "alpha", 1: "train", 2: "test"})
    errors_df.set_index("alpha", inplace=True)
    tidy = errors_df.stack().reset_index().rename(columns={"level_1": "type", 0: "errors"})
    # plot the courbes

    plt.figure(figsize=(10, 5))
    ax = plt.gca()
    ax.grid(True)
    sns.lineplot(data=tidy, x='alpha', y='errors', hue='type')
    plt.show()


if __name__ == '__main__':
    train, test = get_vowel_datsets()
    print(train.shape)
    print(test.shape)

    x_train = norm(train[:, 1:])
    y_train = train[:, 0]
    x_test = norm(test[:, 1:])
    y_test = test[:, 0]

    calc_linear_regression(x_train, y_train, x_test, y_test)
    calc_lda(x_train, y_train, x_test, y_test)
    calc_qda(x_train, y_train, x_test, y_test)
    calc_logistic(x_train, y_train, x_test, y_test)
    create_graph(x_train, y_train, x_test, y_test)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv("../datasets/phoneme/phoneme.data", index_col=0)
print(data.head())
print(data.shape)

#Extracting all "aa" phonemes
aa_data = data[data['g']=='aa']
#Taking a sample of 15 examples
aa_sample = aa_data.sample(15)
#same for "ao"
ao_data = data[data['g']=='ao']
ao_sample = ao_data.sample(15)
#concat dataframes
df = pd.concat([aa_data, ao_data]).reset_index(drop=True)
#selecting features
features = df.columns[:-2]
#train and test split
train = df.sample(frac=0.8,random_state=1)
test = df.drop(train.index)
x_train = train[features]
x_test = test[features]
y_train = train['g']
y_test = test['g']

#Logistic regression
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=100000000, penalty="none")
reg = model.fit(x_train, y_train)
#print coefficients
params = reg.coef_
#print(params)

#get score
print("Raw datas:")
print(F"Error score for train:{1 - reg.score(x_train, y_train)}")
print(F"Error score for test:{1 - reg.score(x_test, y_test)}")

#generating a spline basis
from patsy import dmatrices, dmatrix
x = range(1,257)
spline_basis = dmatrix("cr(x, df=12) - 1", {"x": x}, return_type='dataframe')
#print(spline_basis)
#filtering data by spline
x_train_spline = np.dot(x_train, spline_basis)
x_test_spline = np.dot(x_test, spline_basis)

#Logistic regression with spline
from sklearn.linear_model import LogisticRegression
model_spline = LogisticRegression(max_iter=100000000, penalty="none")
reg_spline = model_spline.fit(x_train_spline, y_train)
#print coefficients
params_spline = reg_spline.coef_
#print(params_spline)
#get full curve
params_filtered = np.dot(spline_basis, params_spline.T)
#print(params_filtered)
#get score
print("With spline:")
print(F"Error score for train:{1 - reg_spline.score(x_train_spline, y_train)}")
print(F"Error score for test:{1 - reg_spline.score(x_test_spline, y_test)}")
#plot the graphs
fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(10,11))
for i in range(ao_sample.shape[0]):
    y = ao_sample.iloc[i,:-2]
    x = range(len(y))
    if i==0:
        ax[0].plot(x, y, color='orange', linewidth=0.3, label='ao')
    else:
        ax[0].plot(x, y, color='orange', linewidth=0.3)
for i in range(aa_sample.shape[0]):
    y = aa_sample.iloc[i,:-2]
    x = range(len(y))
    if i==0:
        ax[0].plot(x, y, color='green', linewidth=0.3, label='aa')
    else:
        ax[0].plot(x, y, color='green', linewidth=0.3)
    ax[0].legend(bbox_to_anchor=(0.75, 1), loc='upper left')
ax[1].plot(x, params.T, color='gray', linewidth=0.5)
ax[1].plot(x, params_filtered, color='red', linewidth=1)
ax[1].axhline(y=0, color='k', lw='0.5')
ax[1].set_xlabel("Frequency", fontsize=12)
ax[0].set_xlabel("Frequency", fontsize=12)
ax[1].set_ylabel("Logistic Regression Coefficients", fontsize=12)
ax[0].set_ylabel("Log-periodogram", fontsize=12)
ax[0].set_title("Phoneme Examples", fontsize=15)
ax[1].set_title("Phoneme Classification: Raw and Restricted Logistic Regression", fontsize=15)

plt.subplots_adjust(hspace=0.4)
fig.savefig("fig_5.5.png")
plt.show()
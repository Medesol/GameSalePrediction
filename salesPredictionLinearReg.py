from dataCursor import *
from sklearn import linear_model
from sklearn.model_selection import KFold, cross_val_score
import numpy as np
import pandas as pd

if __name__ == '__main__':
    file_name = "data/learning_data.csv"
    origin_data = pd.read_csv(file_name, header=None, encoding='gbk')

    X = []
    y = []
    n_columns = len(origin_data.columns)
    n_rows = len(origin_data)
    for i in range(n_rows):
        y_val = origin_data[0][i]
        x_val = [origin_data[j][i] for j in range(1, n_columns)]
        if y_val < 10:
            X.append(x_val)
            y.append(y_val)

    X = np.array(X)
    y = np.array(y).reshape((-1, 1))

    shuffled_indexes = np.random.permutation(range(X.shape[0]))
    X = X[shuffled_indexes]
    y = y[shuffled_indexes]

    score = []

    kf = KFold(n_splits=5)
    kf.get_n_splits(X)
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        reg = linear_model.LinearRegression()
        reg.fit(X_train, y_train)

        y_pre = reg.predict(X_test)
        score.append(np.average(abs(y_pre - y_test)))

    print(score)
    print(np.average(score))

    scores = cross_val_score(reg, X, y, cv=5)
    print(scores)

    subset = X[[i for i in range(11)]]
    print(reg.predict(subset))


    '''X, sales = dataCursor()
    X = np.array(X)
    na_sales = np.array(sales['na_sales'])
    eu_sales = sales['eu_sales']
    jp_sales = sales['jp_sales']
    other_sales = sales['other_sales']
    global_sales = np.array(sales['global_sales'])

    print("X.shape = ", X.shape)
    print("global_sales.shape = ", global_sales.shape)
    reg = linear_model.LinearRegression()
    reg.fit(X, global_sales)
    params = reg.get_params()
    print("Stop training")
    print(params)'''
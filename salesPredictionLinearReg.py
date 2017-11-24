from dataCursor import *
from sklearn import linear_model
from sklearn.model_selection import KFold, cross_val_score
from sklearn import cross_validation
import numpy as np
import pandas as pd

if __name__ == '__main__':
    file_name = "data/merging_output.csv"
    origin_data = pd.read_csv(file_name, encoding='gbk')

    #X = np.array([x for x in origin_data['score'] if x < 10])
    #y = np.array(origin_data['Global_Sales'])

    X = []
    y = []
    for i in range(len(origin_data['score'])):
        x_val = origin_data['score'][i]
        y_val = origin_data['Global_Sales'][i]
        if y_val < 10:
            X.append(x_val)
            y.append(y_val)

    X = np.array(X).reshape(-1, 1)
    y = np.array(y)

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
    score = np.average(score)

    scores = cross_val_score(reg, X, y, cv=5)
    print(score)


    for i in range(11):
        print(reg.predict(i))


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
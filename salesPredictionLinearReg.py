from dataCursor import *
from pprint import pprint
from sklearn import linear_model
from sklearn import tree
from sklearn.externals.six import StringIO
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import KFold, cross_val_score
import numpy as np
import pydotplus as pydot
import pandas as pd

def test_linear(X_train, y_train, X_test, y_test):

    # Linear regression prediction
    discretize = lambda y : np.array([0 if x < 1 else 1 if x < 5 else 2 for x in y])
    discrete_y_train = discretize(y_train)
    discrete_y_test = discretize(y_test)
    reg = linear_model.LinearRegression()
    reg.fit(X_train, discrete_y_train)
    y_pre = reg.predict(X_test)
    errors = abs(y_pre - discrete_y_test)

    return errors

def test_lda(X_train, y_train, X_test, y_test):
    # LDA analysis
    #discrete_y = (y_train * 2).astype(int).reshape(y_train.shape[0])
    discretize = lambda y : np.array([0 if x < 1 else 1 if x < 5 else 2 for x in y])
    discrete_y_train = discretize(y_train)
    discrete_y_test = discretize(y_test)

    lda = LinearDiscriminantAnalysis(n_components=1)
    lda.fit(X_train, discrete_y_train)
    y_pred = lda.predict(X_test)

    errors = abs(y_pred - discrete_y_test)

    return errors

def test_decisiontree(X_train, y_train, X_test, y_test):
    # Decision Tree Classifier
    discretize = lambda y : np.array([0 if x < 1 else 1 if x < 5 else 2 for x in y])
    discrete_y_train = discretize(y_train)
    discrete_y_test = discretize(y_test)

    clf = tree.DecisionTreeClassifier(criterion="entropy")
    clf.fit(X_train, discrete_y_train)
    dot_data = StringIO()
    tree.export_graphviz(clf, out_file=dot_data, filled=True, rounded=True)
    graph = pydot.graph_from_dot_data(dot_data.getvalue())
    graph.write_pdf('Tree_Graph.pdf')
    print('Check file Tree_graph.pdf for Decision Tree information')
    print(clf.score(X_test, discrete_y_test))
    y_predict = clf.predict(X_test)
    errors = abs(y_predict - discrete_y_test)
    return errors

def evaluate(test_model):
    scores = []

    # Amount of errors falling between previous_index/10 and index/10
    # I.E: spectrum[1] = amount of errors between 0 and 0.1 millions off
    spectrum = {}

    kf = KFold(n_splits=5)
    kf.get_n_splits(X)
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        errors = test_model(X_train, y_train, X_test, y_test)

        scores.append(np.average(errors))

        for error in errors:
            index = int(error * 10) / 10 #int(error * 10)
            if index in spectrum:
                spectrum[index] += 1
            else:
                spectrum[index] = 1

    print("Error averages per test fold:")
    print(scores)
    print("Error average:")
    print(np.average(scores))
    print("Error frequencies:")
    pprint(spectrum)

    #scores = cross_val_score(model, X, y, cv=5)
    #print(scores)

    #subset = X[[i for i in range(11)]]
    #print(model.predict(subset))


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

    print("Linear regression:")
    evaluate(test_linear)

    print("\n\nLDA:")
    evaluate(test_lda)

    print("\n\nDecision Tree:")
    evaluate(test_decisiontree)

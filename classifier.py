import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier

TEST_SIZE = 0.33
LABELS = ["Brian", "Eleanor"]
FILES = ["brian_keylogs_clean.csv", "eleanor_keylogs_clean.csv"]
top_pairs = None

def featurize_that_bish(df, top_pairs=None):
    data = pd.DataFrame(df.groupby("pair")['delta'].mean().reset_index())
    if top_pairs:
        data = data.where(data["pair"].isin(top_pairs)).dropna()
    data.columns = ["pair", "delta_avg"]
    data = data.reset_index()
    del data["index"]
    data = data.sort_values("pair", ascending=False)
    data = data.transpose()
    return data

def combine_data(a, b):
    new = pd.DataFrame(a)
    new.append(b, axis=1)
    return new

def create_matrix(lst):
    mat = lst[0]
    for mat1 in lst[1:]:
        mat = mat.append(mat1)
    return np.asmatrix(mat)

def get_top_pairs():
    return set(dataframes[0].groupby("pair").count().sort_values("delta", ascending=False).reset_index()[:100]['pair'])

def get_train_test(dataframes):
    global top_pairs
    top_pairs = get_top_pairs()
    train_test = [train_test_split(df, test_size = TEST_SIZE) for df in dataframes]
    featurized = [(featurize_that_bish(pair[0], top_pairs), featurize_that_bish(pair[1], top_pairs)) for pair in train_test]
    training_lst = [pair[0].iloc[1:] for pair in featurized]
    test_lst = [pair[1].iloc[1:] for pair in featurized]
    train = np.asmatrix(pd.DataFrame(create_matrix(training_lst)).fillna(0))
    test = np.asmatrix(pd.DataFrame(create_matrix(test_lst)).fillna(0))
    return train, test

def print_probabilities(model, test):
    result = list(model.predict_proba(test))
    print(LABELS)
    for arr in result:
        print(arr)

def train_classifier():
    dataframes = [shuffle(pd.read_csv(file)) for file in FILES]
    train, test = get_train_test(dataframes)
    log = LogisticRegression(penalty='l2')
    log.fit(train, LABELS)
    return log

def classify(clf, averages):
    def format_averages(averages):
        pairs = []
        times = []
        for key in top_pairs:
            delta = 0
            if key in averages:
                delta = averages[key][0]
            pairs.append(key)
            times.append(delta)
        d = {
            "pair" : pairs,
            "delta_avg" : times
        }
        df = pd.DataFrame(data = d)
        df = df[["pair", "delta_avg"]]
        df = df.sort_values("pair", ascending=False)
        df = df.transpose()
        a = df.iloc[1:]
        a = pd.DataFrame(a)
        a = a.fillna(0)
        a = np.asmatrix(a)
        return a
    
    averages = format_averages(averages)
    if averages.shape[1] > 10:
        return clf.predict(averages)
    else:
        return None

dataframes = [shuffle(pd.read_csv(file)) for file in FILES]
train, test = get_train_test(dataframes)


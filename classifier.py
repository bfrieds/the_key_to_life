import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, NuSVC, LinearSVC
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

TEST_SIZE = 0.33
LABELS = ["Brian", "Eleanor", "Walt"]
FILES = ["{0}/{1}_{2}".format("keylog_clean_data", name.lower(), "keylogs_clean.csv") for name in LABELS]
DATAFRAMES = None
TRAIN, TEST, DATA_LABELS = None, None, None
DATAFRAMES = [shuffle(pd.read_csv(file)) for file in FILES]

def featurize_that_bish(df, n, top_pairs=[], frac = 0.01):
    def rand_avg_pairs():
        rand = df.sample(frac=frac)
        data = rand.groupby("pair").mean().reset_index() # random sample here
        if len(top_pairs) > 0:
            data = data.where(data["pair"].isin(top_pairs)).dropna()
        data.columns = ["pair", "delta_avg"]
        data = data.sort_values("pair", ascending=False)
        data = data.transpose()
        data = pd.DataFrame(data)
        data.columns = data.iloc[0]
        data = data.drop(data.index[0])
        return data
    
    final = [pd.DataFrame(columns=top_pairs)]
    for _ in range(n):
        data = rand_avg_pairs()
        final.append(data)
    return pd.DataFrame(create_matrix(final)).fillna(0)

def create_matrix(lst):
    mat = lst[0]
    for mat1 in lst[1:]:
        mat = mat.append(mat1)
    return np.asmatrix(mat)

def format_matrix(mat):
    return np.asmatrix(pd.DataFrame(mat).fillna(0))

def get_top_pairs():
    df = DATAFRAMES[0]
    for df_add in DATAFRAMES[1:]:
        df = df.append(df_add)
    return set(df.groupby("pair").count().sort_values("delta", ascending=False).reset_index()[:100]['pair'])

def get_train_test_labels(dataframes, top_pairs = get_top_pairs(), n = 1000):
    n_train = int(n - TEST_SIZE * n)
    n_test = int(n * TEST_SIZE)
    
    train_test = [train_test_split(df, test_size = TEST_SIZE) for df in dataframes]
    featurized = [(featurize_that_bish(pair[0], n_train, top_pairs), featurize_that_bish(pair[1], n_test, top_pairs)) for pair in train_test]
    train_lst, test_lst = zip(*featurized)
    train = format_matrix(create_matrix(train_lst))
    test = format_matrix(create_matrix(test_lst))
    train_labels = get_labels(n_train)
    test_labels = get_labels(n_test)
    return train, train_labels, test, test_labels

def get_labels(n):
    labels = []
    for label in LABELS:
        labels.extend([label for _ in range(n)])
    return labels
        
def create_metrics(clf, train, train_labels, test, test_labels):
    clf.fit(train, train_labels)
    pred = clf.predict(test)
    return classification_report(test_labels, pred), confusion_matrix(test_labels, pred)

def display_confusion_matrix(matrix, labels):
    df_cm = pd.DataFrame(matrix, labels, labels)
    plt.figure(figsize = (10,7))
    sn.heatmap(df_cm, annot=True, cmap='Blues', fmt='g')
    plt.show()

def display_metrics(clf, train, train_labels, test, test_labels):
    report, matrix = create_metrics(clf, train, train_labels, test, test_labels)
    print(clf.__class__.__name__)
    print(report)
    display_confusion_matrix(matrix, LABELS)
    
def full_send():
    models = [LogisticRegression, SVC, NuSVC, LinearSVC, RandomForestClassifier, AdaBoostClassifier]
    for model in models:
        display_metrics(model(), TRAIN, TRAIN_LABELS, TEST, TEST_LABELS)

def print_probabilities(model, test):
    result = list(model.predict_proba(test))
    print(LABELS)
    for arr in result:
        print(arr)

def train_classifier():
    clf = RandomForestClassifier(bootstrap=False, class_weight=None, criterion='gini',
            max_depth=10, max_features='sqrt', max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=2, min_samples_split=5,
            min_weight_fraction_leaf=0.0, n_estimators=1200, n_jobs=1,
            oob_score=False, random_state=None, verbose=0,
            warm_start=False)
    clf.fit(train, LABELS)
    return clf

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
    print(averages.shape)
    if averages.shape[1] > 10:
        return clf.predict(averages)
    else:
        return None
print("yay")
TRAIN, TRAIN_LABELS, _, _ = get_train_test_labels(DATAFRAMES)


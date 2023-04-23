from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, ConfusionMatrixDisplay, classification_report
from sklearn import metrics
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA

#Optimal Estimators: 1400
#Optimal Depth: 16
#Optimal Eta: 0.01
#Best accuracy: 1.0

def optimize():
    x = pd.read_csv('x.csv')
    y = pd.read_csv('y.csv')['label']
    encoder = LabelEncoder()
    y = encoder.fit_transform(y)

    scaler = MinMaxScaler()
    x = scaler.fit_transform(x)

    search = {'n_estimators': np.arange(900, 1500, 100), 'max_depth': [3, 8, 16], 'eta':[.1, .01, .001], 'subsample':[.5]}
    grid = RandomizedSearchCV(XGBClassifier(), search, cv=5, scoring='f1_micro', n_iter=5)
    grid.fit(x, y)

    print("Optimal Estimators: " + str(grid.best_params_['n_estimators']))
    print("Optimal Depth: " + str(grid.best_params_['max_depth']))
    print("Optimal Eta: " + str(grid.best_params_['eta']))

    yHat = grid.predict(x)
    print("Best accuracy: " + str(accuracy_score(y, yHat)))
    print("Best F1: " + str(grid.best_score_))
    
def main():
    labels = ['walking', 'shuffling', 'ascending', 'descending', 'standing', 'sitting', 'lying']
    x = pd.read_csv('x.csv').to_numpy()
    y = pd.read_csv('y.csv')['label'].to_numpy()
    encoder = LabelEncoder()
    y = encoder.fit_transform(y)

    pca = PCA(n_components='mle')
    x = pca.fit_transform(x)

    acc_test = 0

    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)

    for train, test in cv.split(x, y):
        xTrain = x[train]
        xTest = x[test]
        yTrain = y[train]
        yTest = y[test]
        
        scaler = MinMaxScaler()
        xTrain = scaler.fit_transform(xTrain)
        xTest = scaler.transform(xTest)

        model = XGBClassifier(n_estimators=1400, eta=0.01, max_depth=16, subsample=.5, objective='multi:softmax')
        model.fit(xTrain, yTrain)
        
        yTestHat = model.predict(xTest)
        print(classification_report(yTest, yTestHat, target_names=labels))

        acc_test += accuracy_score(yTest, yTestHat)


    acc_test /= cv.get_n_splits(x)
    
    print("Testing Accuracy: " + str(acc_test))
    
if __name__ == "__main__":
    main()

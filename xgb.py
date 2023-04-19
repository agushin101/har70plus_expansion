from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, ConfusionMatrixDisplay
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

#Optimal Estimators: 475
#Optimal Depth: 10
#0.945
#https://xgboost.readthedocs.io/en/stable/parameter.html

def optimize():
    x = pd.read_csv('x.csv')
    y = pd.read_csv('y.csv')['label']
    encoder = LabelEncoder()
    y = encoder.fit_transform(y)

    xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.2)

    scaler = MinMaxScaler()
    xTrain = scaler.fit_transform(xTrain)
    xTest = scaler.transform(xTest)

    search = {'n_estimators': np.arange(400, 501, 15), 'max_depth': np.arange(10, 200, 40)}
    grid = GridSearchCV(XGBClassifier(), search, cv=2, scoring='f1_micro')
    grid.fit(xTrain, yTrain)

    print("Optimal Estimators: " + str(grid.best_params_['n_estimators']))
    print("Optimal Depth: " + str(grid.best_params_['max_depth']))

    yHat = grid.predict(xTest)
    print(accuracy_score(yTest, yHat))
    
def main():
    x = pd.read_csv('x.csv')
    y = pd.read_csv('y.csv')['label']
    encoder = LabelEncoder()
    y = encoder.fit_transform(y)

    xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.2)

    scaler = MinMaxScaler()
    xTrain = scaler.fit_transform(xTrain)
    xTest = scaler.transform(xTest)

    model = XGBClassifier(n_estimators=1024, eta=0.1, max_depth=6, reg_lambda=1)
    model.fit(xTrain, yTrain)
    yHat = model.predict(xTest)
    confusion = confusion_matrix(yTest, yHat)
    disp = ConfusionMatrixDisplay(confusion)
    disp.plot()
    plt.show()
    print(accuracy_score(yTest, yHat))
    
if __name__ == "__main__":
    main()

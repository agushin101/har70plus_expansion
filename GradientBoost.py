import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from matplotlib.colors import ListedColormap
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score
from sklearn import metrics
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn import utils
import numpy as np


def main():

#reading the datasets 
    df_features = pd.read_csv('x.csv')
    df_labels = pd.read_csv('y.csv')

    # Train test split using sklearn
    
    X_train, X_test, y_train, y_test = train_test_split(
       df_features , df_labels['label'], test_size=0.2, 
    )
    # standardize the features using MinMaxScaler
    sc = StandardScaler()
    X_train_std = sc.fit_transform(X_train)
    X_test_std = sc.transform(X_test)

    # creating the models
    pca = PCA(n_components='mle')
    model = GradientBoostingClassifier()

    #convert y values to categorical values
    lab = preprocessing.LabelEncoder()
    y_transformed = lab.fit_transform(y_train)
    y_transformed1 = lab.fit_transform(y_test)

    #adding PCA
    X_train_pca = pca.fit_transform(X_train_std)
    X_test_pca = pca.transform(X_test_std)

    # model with pca
    model.fit(X_train_pca ,y_transformed)
    y_pred= model.predict(X_test_pca)
    acc = accuracy_score(y_transformed1, y_pred)
    print()
    print("Metrics of Gradient Boost Classifier with PCA-------------")
    print()
    print("Accuarcy:")
    print(acc)
    print()
    print("Confusion Matrix:")
    confusion_matrix = metrics.confusion_matrix(y_transformed1, y_pred)
    print(confusion_matrix)
    print()
  

    #model without pca
    model.fit(X_train_std ,y_transformed)
    y_pred= model.predict(X_test_std)
    acc = accuracy_score(y_transformed1, y_pred)
    print()
    print("Metrics of Gradient Boost Classifier without PCA-------------")
    print()
    print("Accuarcy:")
    print(acc)
    print()
    print("Confusion Matrix:")
    confusion_matrix = metrics.confusion_matrix(y_transformed1, y_pred)
    print(confusion_matrix)
    print()


if __name__ == "__main__":
    main()

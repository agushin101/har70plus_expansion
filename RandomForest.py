import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from matplotlib.colors import ListedColormap
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Lasso
from sklearn import utils
import numpy as np


def main():

#reading the datasets 
    df_features = pd.read_csv('processing/x.csv')
    df_labels = pd.read_csv('processing/y.csv')

    # Train test split using sklearn
    
    X_train, X_test, y_train, y_test = train_test_split(
       df_features , df_labels['label'], test_size=0.2, 
    )
    # standardize the features using MinMaxScaler
    sc = MinMaxScaler()
    X_train_std = sc.fit_transform(X_train)
    X_test_std = sc.transform(X_test)

    # use Lasso for feature selection
    lasso = Lasso(alpha=0.01, random_state=0)
    sel = SelectFromModel(lasso)
    sel.fit(X_train_std, y_train)

    # get the selected features
    selected_features = X_train.columns[sel.get_support()]
    X_train_sel = sel.transform(X_train_std)
    X_test_sel = sel.transform(X_test_std)


    # creating the models
    pca = PCA(n_components='mle')
    model = RandomForestClassifier()


    

    #convert y values to categorical values
    lab = preprocessing.LabelEncoder()
    y_transformed = lab.fit_transform(y_train)
    y_transformed1 = lab.fit_transform(y_test)

    #adding PCA
    X_train_pca = pca.fit_transform(X_train_sel)
    X_test_pca = pca.transform(X_test_sel)


    param_grid = {
        'n_estimators': [50, 100, 200],
    }

    # perform grid search with 5-fold cross validation
    grid_search = GridSearchCV(model, param_grid, cv=5)
    grid_search.fit(X_train_pca, y_transformed)

    # model with pca
    print()
    print("Metrics of Random Forest Classifier with PCA-------------")
    print()
    print('Best Hyperparameters:', grid_search.best_params_)
    y_pred = grid_search.predict(X_test_pca)
    acc = accuracy_score(y_transformed1, y_pred)
    print('Accuracy:', acc)
    print('Confusion Matrix:')
    confusion_matrix = metrics.confusion_matrix(y_transformed1, y_pred)
    print(confusion_matrix)
    print()




    #model without pca
    grid_search = GridSearchCV(model, param_grid, cv=5)
    grid_search.fit(X_train_sel, y_transformed)
    print()
    print("Metrics of Random Forest Classifier without PCA-------------")
    print()
    print('Best Hyperparameters:', grid_search.best_params_)
    y_pred = grid_search.predict(X_test_sel)
    acc = accuracy_score(y_transformed1, y_pred)
    print('Accuracy:', acc)
    print('Confusion Matrix:')
    confusion_matrix = metrics.confusion_matrix(y_transformed1, y_pred)
    print(confusion_matrix)
    print()

    


if __name__ == "__main__":
    main()

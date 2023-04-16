from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt


def knn_gridsearch(X_train, y_train, X_test, y_test):
    #performing Min-Max scaling on the training and testing data
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    #KNN classifier created
    knn = KNeighborsClassifier()

    #the range of k values to test; no even k values
    k_values = list(range(1, 102, 2))

    #parameter grid to search over created 
    param_grid = {'n_neighbors': k_values}

    #GridSearchCV object created
    grid_search = GridSearchCV(knn, param_grid, cv=5)

    #GridSearchCV object fit to the scaled training data
    grid_search.fit(X_train_scaled, y_train)

    #plot results of GridSearchCV
    plot_gridsearch(grid_search)

    #obtaining best K value from the gridSearch
    best_k = grid_search.best_params_['n_neighbors']

    #new KNN classifier obj with the best k value created
    knn_best = KNeighborsClassifier(n_neighbors=best_k)

    #new KNN classifier trained on the scaled training data
    knn_best.fit(X_train_scaled, y_train)

    #class labels for the scaled testing data predicted
    y_pred = knn_best.predict(X_test_scaled)

    #print the accuracy of the model; print best k value (determined using grid_search.best_params_)
    print("Best K value:", best_k)
    print("Accuracy:", knn_best.score(X_test_scaled, y_test))

    return knn_best

def plot_gridsearch(grid_search):
    k_values = list(range(1, 102, 2))
    mean_scores = grid_search.cv_results_['mean_test_score']
    plt.plot(k_values, mean_scores)
    plt.xlabel('K Value')
    plt.ylabel('Mean Test Score')
    plt.title('KNN Parameter Optimization')
    plt.show()


if __name__ == "__main__":
    #loading datasets as a dataframe
    df_features = pd.read_csv('processing/features/1.csv')
    #df_labels = pd.read_csv('processing/labels/1_y.csv')
    df_labels = pd.read_csv('processing/final_labels/1yf.csv')

    #print(df_features.shape)
    #print(df_labels.shape)

    #setting seed value for reproducibility
    seed_value = 124

    #dataset split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(df_features, df_labels, test_size=0.2, random_state=seed_value)#0.4)
    knn_gridsearch(X_train, y_train, X_test, y_test)


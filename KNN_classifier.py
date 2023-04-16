from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import pandas as pd

#loading datasets as a dataframe
df_features = pd.read_csv('processing/features/1.csv')
#df_labels = pd.read_csv('processing/labels/1_y.csv')
df_labels = pd.read_csv('processing/final_labels/1yf.csv')

#print(df_features.shape)
#print(df_labels.shape)

#setting seed value for reproducibility
seed_value = 123#81#124

#dataset split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df_features, df_labels, test_size=0.2, random_state=seed_value)#0.4)
#40% of the dataset will be used for testing, and the remaining 60% will be used for training

#KNN classifier created with k=3
knn = KNeighborsClassifier(n_neighbors=3)

#model trained on the training data
knn.fit(X_train, y_train)

#class labels predicted for the testing data
y_pred = knn.predict(X_test)

#printing acc of the model
print("Accuracy:", knn.score(X_test, y_test)) #y_pred

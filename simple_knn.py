#Imported Libraries
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from mlxtend.plotting import plot_decision_regions

#Splitting the data into train and test
def split_input_data(input_data):
    X = input_data.iloc[:,:-1]
    y = input_data.iloc[:,-1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    return X_train, X_test, y_train, y_test

#Displaying decision regions 
def display_contours(n_neighbors, clf, X_train, y_train):
    X_train1 = X_train.to_numpy()
    y_train1 = y_train.to_numpy()
    plot_decision_regions(X_train1, y_train1, clf=clf, legend=2)
    plt.xlabel("ParamA")
    plt.ylabel("ParamB")
    plt.title("KNN with K=" + str(n_neighbors))
    plt.show()

#Applying KNeighbors Classifier to data
def knn(n_neighbors, X_train, y_train, X_test): 
    clf = KNeighborsClassifier(n_neighbors)
    clf.fit(X_train, y_train)
    predicted_y = clf.predict(X_test)
    display_contours(n_neighbors, clf, X_train, y_train)
    return predicted_y

#Performance of KNN classification algorithm
def evaluateknn(y_predicted, y_test):
    print(confusion_matrix(y_test, predicted_y))
    print(classification_report(y_test, predicted_y))

#Can Change the number of neighbors below in the KNN function below
if __name__ == "__main__":
    input_data = pd.read_csv("inputData.csv")
    X_train, X_test, y_train, y_test = split_input_data(input_data)
    predicted_y = knn(3, X_train, y_train, X_test)
    evaluateknn(predicted_y, y_test)

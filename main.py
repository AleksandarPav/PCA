import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer # dataset with breast cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report


def main():
    # the goal is to perform Principal Component Analysis and to select only 2 most important components

    # data is read from sklearn's breast cancer dataset
    cancer = load_breast_cancer()
    print(cancer.keys()) # cancer acts like a dictionary
    print(cancer['DESCR']) # 30 attributes in dataset

    # reading data into a dataframe
    df = pd.DataFrame(cancer['data'], columns = cancer['feature_names'])
    print(df.head())

    # scaling the data
    scaler = StandardScaler()
    scaler.fit(df)
    scaled_data = scaler.transform(df) # z-normalization of df data

    # PCA
    pca = PCA(n_components = 2) # number of the most significant components to keep
    pca.fit(scaled_data)
    x_pca = pca.transform(scaled_data)

    # the number of features went from 30 to 2:
    print(scaled_data.shape, '\n', x_pca.shape) # 569x30, 569x2

    # coloring data based on their belonging to 'malignant' or 'benign'; it can be seen that the data is very well
    # linearly separated based on the two components
    plt.figure(figsize = (10, 6))
    plt.scatter(x_pca[:, 0], x_pca[:, 1], c = cancer['target'], cmap = 'plasma')
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')

    # printing principal components relationship to the original features from the dataset
    print(pca.components_) # 2x30; 2 principal components, 30 original features

    # performing Logistic Regression to see how well was the PCA
    X = x_pca
    y = cancer['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

    logReg = LogisticRegression()
    logReg.fit(X_train, y_train)

    predictions = logReg.predict(X_test)

    # classification report and confusion matrix for evaluating predictions
    print(classification_report(y_test, predictions))
    print(confusion_matrix(y_test, predictions))

    # plotting both malignant and benign predictions, separated with color
    plt.figure(figsize = (10, 6))
    plt.scatter(X_test[:, 0], X_test[:, 1], c = predictions, cmap = 'plasma')

    plt.show()


if __name__ == '__main__':
    main()
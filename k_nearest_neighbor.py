import numpy as np
from collections import Counter
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

def find_optimal_k(X_train, y_train, max_k = 100):
    k_list = []
    for i in range(max_k):
        if i % 2 != 0:
            k_list.append(i)

    cv_scores = []

    for k in k_list:
        knn = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn, X_train, y_train, cv = 10, scoring = 'accuracy')
        cv_scores.append(scores.mean())

    mse = [1 - x for x in cv_scores]

    optimal_k = k_list[mse.index(min(mse))]
    print('Using optimal K value of: ' + str(optimal_k))

    knn = KNeighborsClassifier(n_neighbors=optimal_k)
    knn.fit(X_train, y_train)
    y_hat = knn.predict(X_train)
    print('Accuracy score: ' + str(accuracy_score(y_train, y_hat)))

    plt.plot(k_list, mse)
    plt.xlabel('Number of Neighbors K')
    plt.ylabel('Misclassification Error')
    plt.show()

    return optimal_k


def predict(X_train, y_train, x_test, k):
    distances = []
    targets = []

    for i in range(len(X_train)):
        distance = np.sqrt(np.sum(np.square(np.subtract(x_test, X_train[i]))))
        distances.append([distance, i])

    distances = sorted(distances)

    for i in range(k):
        index = distances[i][1]
        targets.append(y_train[index])

    prediction = Counter(targets).most_common(1)[0][0]
    return prediction


def k_nearest_neighbor(X_train, y_train, x_test, k):
    prediction = predict(X_train, y_train, x_test, k)
    return prediction

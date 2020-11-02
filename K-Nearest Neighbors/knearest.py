'''
LIBRARIES
'''
import pandas as pd
from sklearn import neighbors, metrics, preprocessing, model_selection

'''
LOAD DATABASE
'''
url_data = "https://drive.google.com/u/0/uc?id=1i9fo-zNDTpr9R6XdbqoF8vOgyA03xZtl&export=download"
dataset = pd.read_csv(url_data)

'''
SELECTION OF INPUT AND OUTPUT ATTRIBUTES
(First 10 input and the last output)
Review this if you want to change the
database!!
'''
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:,10].values

'''
PART OF 20% OF THE DATABASE FOR TESTS
'''
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=0.20)

'''
NORMALIZATION OF ATTRIBUTES
'''
scaler = preprocessing.RobustScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

'''
AUXILIARY FUNCTION TO ASSIST TESTS
'''
def knn(k, metric="euclidean"):
  knn = neighbors.KNeighborsClassifier(n_neighbors=k, metric=metric)
  knn.fit(X_train, Y_train)
  Y_prediction = knn.predict(X_test)
  return Y_prediction

k, metric = input("Input a k value and the distance metric(ex: 5 euclidean): ").split()
Y_prediction = knn(int(k), metric)

print("\nK: " + str(k) + "\nMetric: " + metric)
print('\nConfusion Matrix')

print(metrics.confusion_matrix(Y_test, Y_prediction))
print('\nClassification Report')
print(metrics.classification_report(Y_test, Y_prediction))

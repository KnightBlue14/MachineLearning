import pickle
from sklearn import linear_model
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

digits = datasets.load_digits()
X = digits.data
y = digits.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

model = linear_model.LogisticRegression(max_iter = 10000).fit(X_train, y_train)

pck_file = "example_Model.pkl"
with open(pck_file, 'wb') as file:  
    pickle.dump(model, file)
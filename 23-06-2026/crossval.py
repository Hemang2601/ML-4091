from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

# 1. Load the dataset
iris = load_iris()

# 2. Fix: Increase max_iter to 1000 to allow the lbfgs solver to converge safely
logreg = LogisticRegression(max_iter=1000)

# 3. Three-fold cross-validation
scores_3 = cross_val_score(logreg, iris.data, iris.target, cv=3)
print("three Cross-validation scores: {}".format(scores_3))
print("Average cross-validation score: {:.2f}\n".format(scores_3.mean()))

# 4. Five-fold cross-validation
scores_5 = cross_val_score(logreg, iris.data, iris.target, cv=5)
print("five Cross-validation scores: {}".format(scores_5))
print("Average cross-validation score: {:.2f}".format(scores_5.mean()))
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut

# Load dataset
iris = load_iris()
print("Iris labels:\n{}".format(iris.target))

# FIX: Added max_iter=1000 to allow the solver to converge
logreg = LogisticRegression(max_iter=1000)

# Setup LOOCV
loo = LeaveOneOut()
scores = cross_val_score(logreg, iris.data, iris.target, cv=loo)

print("\n--- Results ---")
print("Number of cv iterations: ", len(scores))
print("Mean accuracy: {:.2f}".format(scores.mean()))
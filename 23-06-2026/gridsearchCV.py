# naive grid search implementation
from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import pandas as pd
from IPython.display import display
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mglearn.plots

iris = load_iris()

param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100], 'gamma': [0.001, 0.01, 0.1, 1, 10, 100]}
print("Parameter grid:\n{}".format(param_grid))

grid_search = GridSearchCV(SVC(), param_grid, cv=5)
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=0)

grid_search.fit(X_train, y_train)

print("Test set score: {:.2f}".format(grid_search.score(X_test, y_test)))

print("Best parameters: {}".format(grid_search.best_params_))
print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))

print("Best estimator:\n{}".format(grid_search.best_estimator_))

# convert to DataFrame
results = pd.DataFrame(grid_search.cv_results_)
# show the first 5 rows
display(results.head())

# Reshape the cross-validation scores into a 2D grid (6x6)
scores = np.array(results.mean_test_score).reshape(6, 6)

# Plot using Seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(
    data=scores,
    annot=True,
    fmt=".3f",
    xticklabels=param_grid['gamma'],
    yticklabels=param_grid['C'],
    cmap="viridis"
)

# Add titles and axis labels
plt.title("Grid Search CV Mean Test Scores")
plt.xlabel("gamma")
plt.ylabel("C")
plt.show()

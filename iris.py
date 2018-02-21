import pandas as pd
import sklearn as sk
import numpy as np
import sklearn.preprocessing as skp
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data", sep=',', names=['sepel_length', 'sepel_width', 'petal_length', 'petal_width', 'class'])

lb = skp.LabelEncoder()
df['class'] = lb.fit_transform(df['class'])
X, y = df.drop(['class'], axis=1), df['class']


outer_cv = StratifiedKFold(10)
inner_cv = StratifiedKFold(10)


pipe = Pipeline([
    ('clf', SVC()),
    ])

params = [{
            'clf': (SVC(),),
            'clf__C' : [1, 10, 100],
            'clf__kernel': ['poly', 'sigmoid', 'rbf'],
        }, {
            'clf': (LogisticRegression(),),
        }, {
            'clf': (KNeighborsRegressor(),),
            'clf__n_neighbors': [3, 5, 7],
            'clf__p': [1, 2],
        }, {
            'clf': (RandomForestClassifier(),),
            'clf__n_estimators': [100, 500, 1000],
            'clf__max_features': [.3, .7],
        }, {
            'clf': (AdaBoostClassifier(),),
            'clf__n_estimators': [50, 100, 500],
            'clf__learning_rate': [.1, 1.0, 3.0],
            }]

clf = GridSearchCV(pipe, param_grid=params, cv=inner_cv, verbose=1)
clf.fit(X, y)

scores = cross_val_score(clf, X, y, cv=outer_cv, scoring='accuracy', verbose=3)





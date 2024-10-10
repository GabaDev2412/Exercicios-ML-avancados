import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from tpot import TPOTClassifier

# Criar um conjunto de dados sintético
X, y = make_classification(n_samples=1000, n_features=10, n_informative=8, n_redundant=2,
                           n_clusters_per_class=1, random_state=42)

data = pd.DataFrame(X, columns=[f'feature_{i+1}' for i in range(X.shape[1])])
data['target'] = y

data.to_csv('synthetic_data.csv', index=False)

tpot_data = pd.read_csv('synthetic_data.csv', dtype=np.float64)
features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'], random_state=42)

X_train, X_test, y_train, y_test = train_test_split(data.drop('target', axis=1), 
                                                    data['target'], 
                                                    test_size=0.2, 
                                                    random_state=42)

dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
dt_predictions = dt_model.predict(X_test)
dt_accuracy = accuracy_score(y_test, dt_predictions)

rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_predictions)

print(f'Accuracy of Decision Tree: {dt_accuracy:.2f}')
print(f'Accuracy of Random Forest: {rf_accuracy:.2f}')

tpot = TPOTClassifier(verbosity=2, random_state=42, generations=5, population_size=20)
tpot.fit(training_features, training_target)

tpot_predictions = tpot.predict(testing_features)
tpot_accuracy = accuracy_score(testing_target, tpot_predictions)

print(f'Accuracy of TPOT: {tpot_accuracy:.2f}')

tpot.export('best_model_pipeline.py')

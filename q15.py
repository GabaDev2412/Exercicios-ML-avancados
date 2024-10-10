import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

file_path = 'top_rated_9000_movies_on_TMDB.csv'
data = pd.read_csv(file_path)

data['vote_class'] = data['vote_average'].apply(lambda x: 1 if x >= 7 else 0)

numerical_cols = ['vote_average', 'vote_count', 'popularity']
X = data[numerical_cols]
y = data['vote_class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pipeline = Pipeline([
    ('scaler', StandardScaler()),  
    ('pca', PCA(n_components=2)),  
    ('classifier', RandomForestClassifier(random_state=42)) 
])

cross_val_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='accuracy')

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)

test_accuracy = accuracy_score(y_test, y_pred)

print(f"Cross-validation Accuracy Scores: {cross_val_scores}")
print(f"Mean CV Accuracy: {cross_val_scores.mean():.2f}")
print(f"Test Set Accuracy: {test_accuracy:.2f}")

pca = pipeline.named_steps['pca']
explained_variance = pca.explained_variance_ratio_


plt.figure(figsize=(8, 5))
plt.bar(range(len(explained_variance)), explained_variance, alpha=0.7, align='center')
plt.title('Variância Explicada pelos Componentes do PCA')
plt.xlabel('Componentes Principais')
plt.ylabel('Proporção da Variância Explicada')
plt.grid()
plt.show()

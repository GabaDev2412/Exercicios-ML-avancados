import pandas as pd
import numpy as np
import ast
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import plot_tree

def load_data(file_path):
    """Carrega o conjunto de dados a partir do arquivo CSV."""
    return pd.read_csv(file_path)

def preprocess_data(data):
    """Pré-processa os dados, incluindo a conversão de colunas categóricas."""
    data = data[['vote_average', 'vote_count', 'popularity', 'Genres']].copy()

    data['vote_class'] = data['vote_average'].apply(lambda x: 1 if x >= 7 else 0)

    data['Genres'] = data['Genres'].apply(lambda x: ast.literal_eval(x))

    genres = data['Genres'].explode().unique()
    for genre in genres:
        data[genre] = data['Genres'].apply(lambda genres_list: 1 if genre in genres_list else 0)

    data = data.drop(['vote_average', 'Genres'], axis=1)

    return data.drop('vote_class', axis=1), data['vote_class']

def train_random_forest(X_train, y_train, n_estimators=100, max_depth=None):
    """Treina um modelo de Random Forest e retorna o modelo treinado."""
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)
    return model

def plot_feature_importance(model, feature_names):
    """Plota a importância das características do modelo treinado."""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    plt.figure(figsize=(10, 6))
    plt.title("Importância das Características")
    plt.bar(range(len(importances)), importances[indices], align="center")
    plt.xticks(range(len(importances)), feature_names[indices], rotation=90)
    plt.xlim([-1, len(importances)])
    plt.ylabel('Importância')
    plt.show()

def plot_random_forest_tree(model, feature_names, tree_index=0, max_depth=3):
    """Plota uma árvore individual do modelo Random Forest."""
    plt.figure(figsize=(20, 10))
    estimator = model.estimators_[tree_index]
    plot_tree(estimator, max_depth=max_depth, filled=True, feature_names=feature_names, rounded=True)
    plt.title(f'Árvore {tree_index + 1} (max_depth={max_depth})')
    plt.show()

def main(file_path):
    """Função principal que executa todo o fluxo de trabalho."""
    data = load_data(file_path)
    X, y = preprocess_data(data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf_model = train_random_forest(X_train, y_train, n_estimators=100, max_depth=10)
    plot_feature_importance(rf_model, X.columns)
    plot_random_forest_tree(rf_model, X.columns, tree_index=0, max_depth=3)

if __name__ == "__main__":
    file_path = 'top_rated_9000_movies_on_TMDB.csv'
    main(file_path)

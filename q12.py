import pandas as pd
import numpy as np
import ast
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

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

def train_bagging(X_train, y_train):
    """Treina um modelo de Bagging com base em árvores de decisão."""
    bagging_model = BaggingClassifier(
        estimator=DecisionTreeClassifier(),
        n_estimators=50,
        random_state=42
    )
    bagging_model.fit(X_train, y_train)
    return bagging_model

def train_adaboost(X_train, y_train):
    """Treina um modelo AdaBoost com base em árvores de decisão."""
    ada_model = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=1),
        n_estimators=100,
        random_state=42
    )
    ada_model.fit(X_train, y_train)
    return ada_model

def train_random_forest(X_train, y_train):
    """Treina um modelo de Random Forest."""
    rf_model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
    rf_model.fit(X_train, y_train)
    return rf_model

def train_gradient_boosting(X_train, y_train):
    """Treina um modelo Gradient Boosting."""
    gb_model = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )
    gb_model.fit(X_train, y_train)
    return gb_model

def plot_combined_learning_curves(models, model_names, X, y, title):
    """Plota as curvas de aprendizado de vários modelos em um único gráfico."""
    plt.figure(figsize=(10, 7))
    
    for model, name in zip(models, model_names):
        train_sizes, train_scores, test_scores = learning_curve(model, X, y, cv=5, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10)) 
        train_scores_mean = np.mean(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        plt.plot(train_sizes, test_scores_mean, label=name)
    
    plt.title(title)
    plt.xlabel("Training examples")
    plt.ylabel("Cross-validation score")
    plt.legend(loc="best")
    plt.grid()
    plt.show()

def main(file_path):
    data = load_data(file_path)

    X, y = preprocess_data(data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    bagging_model = train_bagging(X_train, y_train)
    ada_model = train_adaboost(X_train, y_train)
    rf_model = train_random_forest(X_train, y_train)
    gb_model = train_gradient_boosting(X_train, y_train)

    bagging_acc = accuracy_score(y_test, bagging_model.predict(X_test))
    ada_acc = accuracy_score(y_test, ada_model.predict(X_test))
    rf_acc = accuracy_score(y_test, rf_model.predict(X_test))
    gb_acc = accuracy_score(y_test, gb_model.predict(X_test))

    print(f'Acurácia do Bagging: {bagging_acc:.2f}')
    print(f'Acurácia do AdaBoost: {ada_acc:.2f}')
    print(f'Acurácia do Random Forest: {rf_acc:.2f}')
    print(f'Acurácia do Gradient Boosting: {gb_acc:.2f}')

    models = [bagging_model, ada_model, rf_model, gb_model]
    model_names = ['Bagging', 'AdaBoost', 'Random Forest', 'Gradient Boosting']
    plot_combined_learning_curves(models, model_names, X, y, "Comparação de Curvas de Aprendizado - Métodos Ensemble")

if __name__ == "__main__":
    file_path = 'top_rated_9000_movies_on_TMDB.csv'
    main(file_path)

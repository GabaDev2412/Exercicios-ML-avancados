import pandas as pd
import numpy as np
import ast
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
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

def train_decision_tree(X_train, y_train):
    """Treina um modelo de Árvore de Decisão simples."""
    tree_model = DecisionTreeClassifier(random_state=42)
    tree_model.fit(X_train, y_train)
    return tree_model

def train_adaboost(X_train, y_train):
    """Treina um modelo AdaBoost com base em árvores de decisão."""
    ada_model = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=1),
        n_estimators=100,
        random_state=42
    )
    ada_model.fit(X_train, y_train)
    return ada_model

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

def evaluate_model(model, X_test, y_test, model_name):
    """Avalia o modelo e imprime a acurácia."""
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Acurácia do {model_name}: {accuracy:.2f}')
    return accuracy

def plot_comparison(tree_acc, ada_acc, gb_acc):
    """Plota a comparação entre os modelos."""
    models = ['Decision Tree', 'AdaBoost', 'Gradient Boosting']
    accuracies = [tree_acc, ada_acc, gb_acc]
    plt.figure(figsize=(8, 6))
    plt.bar(models, accuracies, color=['skyblue', 'orange', 'green'])
    plt.ylim(0, 1)
    plt.title('Comparação de Acurácia entre Modelos')
    plt.ylabel('Acurácia')
    plt.show()

def main(file_path):
    data = load_data(file_path)

    X, y = preprocess_data(data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    tree_model = train_decision_tree(X_train, y_train)
    ada_model = train_adaboost(X_train, y_train)
    gb_model = train_gradient_boosting(X_train, y_train)

    tree_acc = evaluate_model(tree_model, X_test, y_test, "Decision Tree")
    ada_acc = evaluate_model(ada_model, X_test, y_test, "AdaBoost")
    gb_acc = evaluate_model(gb_model, X_test, y_test, "Gradient Boosting")

    plot_comparison(tree_acc, ada_acc, gb_acc)

if __name__ == "__main__":
    file_path = 'top_rated_9000_movies_on_TMDB.csv'
    main(file_path)

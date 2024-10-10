import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_data(file_path):
    """Carrega o conjunto de dados a partir do arquivo CSV."""
    return pd.read_csv(file_path)

def preprocess_data(data):
    """Pré-processa os dados, selecionando colunas numéricas e codificando rótulos."""
    numerical_data = data[['vote_average', 'vote_count', 'popularity']]
    
    class_labels = data['Genres'].apply(lambda x: x.split(',')[0] if isinstance(x, str) else 'Unknown')
    
    le = LabelEncoder()
    y = le.fit_transform(class_labels)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(numerical_data)
    
    return X_scaled, y, le.classes_

def apply_lda(X, y):
    """Aplica LDA para reduzir a dimensionalidade dos dados."""
    lda = LDA(n_components=2)
    X_lda = lda.fit_transform(X, y)
    return X_lda

def plot_results(X_lda, y, classes):
    """Plota os resultados da LDA."""
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_lda[:, 0], X_lda[:, 1], c=y, cmap='rainbow', alpha=0.7)

    handles, _ = scatter.legend_elements()
    
    plt.legend(handles, classes, title="Classes", bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.xlabel('Componente LDA 1', fontsize=14)
    plt.ylabel('Componente LDA 2', fontsize=14)
    plt.title('Análise Discriminante Linear (LDA) - Separação de Classes', fontsize=16)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def main():
    file_path = 'top_rated_9000_movies_on_TMDB.csv'
    data = load_data(file_path)
    
    if 'Genres' in data.columns:
        X_scaled, y, classes = preprocess_data(data)
        X_lda = apply_lda(X_scaled, y)
        plot_results(X_lda, y, classes)
    else:
        print("A coluna 'Genres' não foi encontrada no conjunto de dados ou não contém classes categóricas utilizáveis.")

if __name__ == "__main__":
    main()

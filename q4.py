import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path)
    numerical_data = data[['vote_average', 'vote_count', 'popularity']]
    
    class_labels = data['Genres'].apply(lambda x: x.split(',')[0] if isinstance(x, str) else 'Unknown')
    le = LabelEncoder()
    y = le.fit_transform(class_labels)
    
    X_scaled = StandardScaler().fit_transform(numerical_data)
    
    return X_scaled, y

def reduce_dimensions(X_train, X_test, n_components=2):
    pca = PCA(n_components=n_components)
    return pca.fit_transform(X_train), pca.transform(X_test)

def evaluate_model(lda, X_test, y_test):
    y_pred = lda.predict(X_test)
    return (accuracy_score(y_test, y_pred), 
            confusion_matrix(y_test, y_pred), 
            classification_report(y_test, y_pred))

def main(file_path):
    X, y = load_and_preprocess_data(file_path)
    
    valid_classes = pd.Series(y).value_counts()[lambda x: x > 1].index
    mask = np.isin(y, valid_classes)
    X, y = X[mask], y[mask]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    X_train_pca, X_test_pca = reduce_dimensions(X_train, X_test)
    
    lda = LDA().fit(X_train_pca, y_train)
    accuracy, cm, report = evaluate_model(lda, X_test_pca, y_test)

    print(f"Precisão do modelo LDA: {accuracy:.2f}")
    print("Matriz de Confusão:")
    print(cm)
    print("\nRelatório de Classificação:")
    print(report)

if __name__ == "__main__":
    main('top_rated_9000_movies_on_TMDB.csv')

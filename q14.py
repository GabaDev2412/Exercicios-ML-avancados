import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

file_path = 'top_rated_9000_movies_on_TMDB.csv'
data = pd.read_csv(file_path)

data['vote_class'] = data['vote_average'].apply(lambda x: 1 if x >= 7 else 0)

numerical_cols = ['vote_average', 'vote_count', 'popularity']
X = data[numerical_cols]
y = data['vote_class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

lda = LDA(n_components=1)
X_train_lda = lda.fit_transform(X_train_scaled, y_train)
X_test_lda = lda.transform(X_test_scaled)

y_train_pred = lda.predict(X_train_scaled)
y_test_pred = lda.predict(X_test_scaled)

train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

print(f'Acurácia no conjunto de treino: {train_accuracy:.2f}')
print(f'Acurácia no conjunto de teste: {test_accuracy:.2f}')

plt.figure(figsize=(8, 5))
plt.scatter(X_train_lda[y_train == 0], np.zeros_like(X_train_lda[y_train == 0]), color='red', label='Ruim (Class 0)', alpha=0.7)
plt.scatter(X_train_lda[y_train == 1], np.ones_like(X_train_lda[y_train == 1]), color='blue', label='Bom (Class 1)', alpha=0.7)
plt.title('Separação das Classes após LDA')
plt.xlabel('Componente Discriminante 1')
plt.ylabel('Classes')
plt.legend()
plt.grid()
plt.show()

lda_coefficients = pd.DataFrame(lda.coef_, columns=numerical_cols)
print("\nCargas das Variáveis nas Componentes Discriminantes:")
print(lda_coefficients)

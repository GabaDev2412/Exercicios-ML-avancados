import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

file_path = 'top_rated_9000_movies_on_TMDB.csv'
data = pd.read_csv(file_path)

numerical_cols = ['vote_average', 'vote_count', 'popularity']
X = data[numerical_cols].copy()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=len(numerical_cols))
X_pca = pca.fit_transform(X_scaled)

explained_variance = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)

for i, var in enumerate(explained_variance):
    print(f"Componente {i+1} explica {var:.2%} da variância.")

print(f"\nVariância explicada acumulada após todos os componentes: {cumulative_variance[-1]:.2%}")

plt.figure(figsize=(8, 5))
plt.bar(range(1, len(explained_variance) + 1), explained_variance, alpha=0.5, align='center', label='Individual Explained Variance')
plt.step(range(1, len(cumulative_variance) + 1), cumulative_variance, where='mid', label='Cumulative Explained Variance')
plt.ylabel('Explained Variance Ratio')
plt.xlabel('Principal Components')
plt.title('PCA - Variância Explicada')
plt.legend(loc='best')
plt.grid()
plt.show()

print("\nCargas dos Componentes Principais:")
components_df = pd.DataFrame(pca.components_, columns=numerical_cols, index=[f'PC{i+1}' for i in range(len(numerical_cols))])
print(components_df)

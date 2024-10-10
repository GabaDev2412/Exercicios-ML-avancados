from sklearn.decomposition import PCA
import seaborn as sns

iris = load_iris()
X = iris.data
y = iris.target
target_names = iris.target_names

pca = PCA(n_components=2)
X_pca_sklearn = pca.fit_transform(X)

explained_variance = pca.explained_variance_ratio_

plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_pca_sklearn[:, 0], y=X_pca_sklearn[:, 1], hue=y, palette='viridis', s=100)
plt.title('Dados Reduzidos com PCA (Scikit-Learn)')
plt.xlabel('Primeiro Componente Principal')
plt.ylabel('Segundo Componente Principal')
plt.grid()
plt.legend(target_names)
plt.show()

explained_variance

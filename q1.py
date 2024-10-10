import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

file_path = 'top_rated_9000_movies_on_TMDB.csv'
data = pd.read_csv(file_path)

numerical_data = data[['vote_average', 'vote_count', 'popularity']]

mean_numerical_data = np.mean(numerical_data, axis=0)
centered_numerical_data = numerical_data - mean_numerical_data

cov_matrix = np.cov(centered_numerical_data.T)

eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

idx = eigenvalues.argsort()[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

eigenvectors_2d = eigenvectors[:, :2]

transformed_data = np.dot(centered_numerical_data, eigenvectors_2d)

fig, axs = plt.subplots(1, 2, figsize=(14, 6))

axs[0].scatter(numerical_data['vote_average'], numerical_data['vote_count'], c='r', marker='o')
axs[0].set_xlabel('Vote Average')
axs[0].set_ylabel('Vote Count')
axs[0].set_title('Dados Originais (Vote Average vs Vote Count)')

axs[1].scatter(transformed_data[:, 0], transformed_data[:, 1], c='b', marker='o')
axs[1].set_xlabel('Componente Principal 1')
axs[1].set_ylabel('Componente Principal 2')
axs[1].set_title('Dados Transformados pelo PCA (2D)')

plt.tight_layout()
plt.show()

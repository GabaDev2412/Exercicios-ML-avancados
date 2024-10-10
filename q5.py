import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

file_path = 'top_rated_9000_movies_on_TMDB.csv'
movies = pd.read_csv(file_path)

print(movies.head())

num_users = 100 
num_movies = len(movies) 
np.random.seed(42)
ratings_matrix = np.random.randint(1, 6, size=(num_users, num_movies))  # Avaliações entre 1 e 5

item_similarity = cosine_similarity(ratings_matrix.T)

def get_recommendations_based_on_item(movie_title, item_similarity, movies, n_recommendations=5):
    movie_idx = movies[movies['title'] == movie_title].index[0]
    similar_scores = list(enumerate(item_similarity[movie_idx]))
    similar_scores = sorted(similar_scores, key=lambda x: x[1], reverse=True)
    similar_scores = similar_scores[1:n_recommendations + 1]
    movie_indices = [i[0] for i in similar_scores]
    return movies.iloc[movie_indices]

recommended_movies = get_recommendations_based_on_item('The Shawshank Redemption', item_similarity, movies)

def plot_recommendations(recommended_movies):
    plt.figure(figsize=(10, 5))
    plt.barh(recommended_movies['title'], recommended_movies['vote_average'], color='skyblue')
    plt.xlabel('Vote Average')
    plt.title('Recommended Movies Based on Item Similarity')
    plt.gca().invert_yaxis()
    plt.show()

plot_recommendations(recommended_movies)

user_similarity = cosine_similarity(ratings_matrix)

def get_recommendations_based_on_user(user_id, user_similarity, ratings_matrix, movies, n_recommendations=5):
    similar_users = list(enumerate(user_similarity[user_id]))
    similar_users = sorted(similar_users, key=lambda x: x[1], reverse=True)
    user_ratings = ratings_matrix[user_id]
    scores = np.zeros(len(movies))
    for i, (similar_user, score) in enumerate(similar_users):
        if i == 0:
            continue
        scores += score * ratings_matrix[similar_user]
    recommended_indices = np.argsort(scores)[-n_recommendations:][::-1]
    return movies.iloc[recommended_indices]

recommended_movies_user = get_recommendations_based_on_user(0, user_similarity, ratings_matrix, movies)

plot_recommendations(recommended_movies_user)

print("\nRecomendações baseadas em item para 'The Shawshank Redemption':")
print(recommended_movies[['title', 'vote_average']])

print("\nRecomendações baseadas em usuário para o usuário 0:")
print(recommended_movies_user[['title', 'vote_average']])

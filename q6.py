import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import matplotlib.pyplot as plt

file_path = 'top_rated_9000_movies_on_TMDB.csv'
movies = pd.read_csv(file_path)

print(movies.head())

movies['Genres'] = movies['Genres'].str.split(',')
movies['content'] = movies['Genres'].apply(lambda x: ' '.join(x)) + ' ' + movies['overview']

tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['content'])

cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

movies = movies.reset_index()
titles = pd.Series(movies['title'])

def get_recommendations(title, cosine_sim=cosine_sim):
    idx = titles[titles == title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    movie_indices = [i[0] for i in sim_scores[1:11]]
    return movies.iloc[movie_indices][['title', 'vote_average', 'vote_count', 'overview']]

movie_title = 'Inception'
recommended_movies = get_recommendations(movie_title)


def plot_recommendations(recommended_movies):
    plt.figure(figsize=(12, 6))
    
    plt.barh(recommended_movies['title'], recommended_movies['vote_average'], color='skyblue')
    
    for index, value in enumerate(recommended_movies['vote_average']):
        plt.text(value, index, f'{value:.2f}', va='center')

    plt.title(f'Recomendações para: {movie_title}')
    plt.xlabel('Média de Votos')
    plt.ylabel('Filmes Recomendados')
    plt.xlim(0, 10)
    plt.show()

plot_recommendations(recommended_movies)

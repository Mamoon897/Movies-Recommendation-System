import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

# Load movie ratings data
ratings = pd.read_csv("https://s3-us-west-2.amazonaws.com/recommender-tutorial/ratings.csv")

# Load movie data
movies = pd.read_csv("https://s3-us-west-2.amazonaws.com/recommender-tutorial/movies.csv")

# Create user-item matrix using scipy csr matrix
def create_matrix(df):
    N = len(df['userId'].unique())
    M = len(df['movieId'].unique())

    # Map Ids to indices
    user_mapper = dict(zip(np.unique(df["userId"]), list(range(N))))
    movie_mapper = dict(zip(np.unique(df["movieId"]), list(range(M))))

    # Map indices to IDs
    user_inv_mapper = dict(zip(list(range(N)), np.unique(df["userId"])))
    movie_inv_mapper = dict(zip(list(range(M)), np.unique(df["movieId"])))

    user_index = [user_mapper[i] for i in df['userId']]
    movie_index = [movie_mapper[i] for i in df['movieId']]

    X = csr_matrix((df["rating"], (movie_index, user_index)), shape=(M, N))

    return X, user_mapper, movie_mapper, user_inv_mapper, movie_inv_mapper

X, user_mapper, movie_mapper, user_inv_mapper, movie_inv_mapper = create_matrix(ratings)

# Find similar movies using kNN
def find_similar_movies(movie_id, X, k, metric='cosine', show_distance=False):
    neighbour_ids = []

    movie_ind = movie_mapper[movie_id]
    movie_vec = X[movie_ind]
    k += 1
    kNN = NearestNeighbors(n_neighbors=k, algorithm="brute", metric=metric)
    kNN.fit(X)
    movie_vec = movie_vec.reshape(1, -1)
    neighbour = kNN.kneighbors(movie_vec, return_distance=show_distance)
    for i in range(0, k):
        n = neighbour.item(i)
        neighbour_ids.append(movie_inv_mapper[n])
    neighbour_ids.pop(0)
    return neighbour_ids


movie_titles = dict(zip(movies['movieId'], movies['title']))

movie_id = 3

similar_ids = find_similar_movies(movie_id, X, k=10)
movie_title = movie_titles[movie_id]

print(f"Since you watched {movie_title}")
print('Recommendations Are')
for i in similar_ids:
    print(movie_titles[i])

# Function to get movieId from movie title
def get_movie_id(movie_title):
    movie_id = movies[movies['title'] == movie_title]['movieId'].values
    if len(movie_id) == 0:
        return None
    return movie_id[0]

# Example usage:
while True:
    input_movie_title = input("Enter the title of the movie you watched (or 'exit' to quit): ")
    if input_movie_title.lower() == 'exit':
        print("Exiting the recommendation system.")
        break

    input_movie_id = get_movie_id(input_movie_title)
    if input_movie_id is None:
        print("Invalid movie title. Please try again with a valid movie title.")
    else:
        similar_ids = find_similar_movies(input_movie_id, X, k=10)
        movie_title = movie_titles[input_movie_id]
        print(f"\nSince you watched {movie_title}\n")
        print("Recommended Movies:")
        for i, movie_id in enumerate(similar_ids, 1):
            recommended_movie_title = movie_titles[movie_id]
            print(f"{i}. {recommended_movie_title}")

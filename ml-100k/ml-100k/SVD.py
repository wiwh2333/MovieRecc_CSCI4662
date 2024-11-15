from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from cluster import Demo_Cluster
import seaborn as sns
from sklearn.manifold import TSNE
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv('u.data', sep='\t', names=['user_id', 'item_id', 'rating', 'timestamp'])

# Drop the timestamp as it's not needed
data = data.drop(columns=['timestamp'])

#Get demograpic clusters (demographic data with cluster column attached)
demo = Demo_Cluster(0)

# Merge the demographic cluster info with the rating data
data = pd.merge(data, demo[['user_id', 'cluster']], on='user_id', how='left')

print(data.head())

# # Create user-item matrix (rows: users, columns: movies)
# user_item_matrix = data.pivot(index='user_id', columns='item_id', values='rating')

# # Fill missing values with 0 (indicating unrated movies)
# user_item_matrix = user_item_matrix.fillna(0)

# # Preview the user-item matrix
# #print(user_item_matrix.head())

# # Apply SVD to the user-item matrix (use a reasonable number of components)
# svd = TruncatedSVD(n_components=50, random_state=42)  # You can adjust n_components
# user_latent_matrix = svd.fit_transform(user_item_matrix)  # User latent matrix (users x latent factors)
# item_latent_matrix = svd.components_  # Item latent matrix (items x latent factors)

# # Check the shape of the matrices
# print("User latent matrix shape:", user_latent_matrix.shape)
# print("Item latent matrix shape:", item_latent_matrix.shape)

# # Merge the user demographic cluster info with the user-item matrix
# user_clusters = demo[['user_id', 'cluster']]  # Assuming 'users' DataFrame has a 'user_cluster' column
# user_clusters = user_clusters.set_index('user_id')

# # # Add the cluster information to the user latent matrix
# cluster_matrix = user_clusters.loc[user_item_matrix.index, 'cluster'].values.reshape(-1, 1)
# user_latent_matrix_with_cluster = np.hstack([user_latent_matrix, cluster_matrix])

# # # # Check the new matrix shape
# print("User latent matrix with cluster shape:", user_latent_matrix_with_cluster.shape)

# # # Predict ratings by taking the dot product of the user and item latent matrices
# predicted_ratings = np.dot(user_latent_matrix_with_cluster, item_latent_matrix.T)

# # Convert predicted ratings into a DataFrame for easier access
# predicted_ratings_df = pd.DataFrame(predicted_ratings, index=user_item_matrix.index, columns=user_item_matrix.columns)

# # For a given user, get the recommended movies (top N movies)
# user_id = 1  # Example user ID
# recommended_movies = predicted_ratings_df.loc[user_id].sort_values(ascending=False)

# # Display the top N recommendations
# top_n = 10  # Show the top 10 recommended movies
# print(f"Top {top_n} recommendations for User {user_id}:")
# print(recommended_movies.head(top_n))
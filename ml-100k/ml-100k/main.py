from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.manifold import TSNE
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from cluster import Demo_Cluster
from cluster import plot_elbow_method



users = Demo_Cluster(1)
print(users.head())
plot_elbow_method(users)


# # t-SNE for 2D projection
# tsne = TSNE(n_components=2, random_state=42)
# tsne_components = tsne.fit_transform(users.drop(['user_id', 'cluster'], axis=1))  # Drop non-features and 'cluster' column

# # Add t-SNE components to the dataframe
# users['tsne1'] = tsne_components[:, 0]
# users['tsne2'] = tsne_components[:, 1]

# # Plotting t-SNE results
# plt.figure(figsize=(8, 6))
# sns.scatterplot(x='tsne1', y='tsne2', hue='cluster', palette='viridis', data=users, s=100, alpha=0.7, edgecolor='k')
# plt.title('t-SNE - User Clusters')
# plt.xlabel('t-SNE Component 1')
# plt.ylabel('t-SNE Component 2')
# plt.legend(title='Cluster')
# plt.show()

# # Loop to print each user ID and its corresponding cluster
# for user_id, cluster_label in zip(users['user_id'], users['cluster']):
#     print(f"User ID: {user_id}, Cluster: {cluster_label}")
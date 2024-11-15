from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.manifold import TSNE
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt




def plot_elbow_method(data, k_range=None):
    """
    Function to plot the Elbow Method for determining the optimal number of clusters.
    
    Parameters:
    - data: The input data for clustering (Pandas DataFrame or NumPy array).
    - k_range: Range of the number of clusters to evaluate. Defaults to range(1, 11).
    """
    # Range of possible cluster values
    k_range = range(1, 11)  # You can test more values of k
    inertia = []

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data)  # Make sure to use the data without non-numeric columns
        inertia.append(kmeans.inertia_)

    plt.plot(k_range, inertia, marker='o')
    plt.title("Elbow Method For Optimal k")
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Inertia")
    plt.show()


def Demo_Cluster(verbose):
    occupations = ["administrator","artist","doctor","educator","engineer","entertainment","executive","healthcare","homemaker","lawyer","librarian","marketing","none","other","programmer","retired","salesman","scientist","student","technician","writer"]
    # Create a dictionary to map occupations to unique numeric values
    occupation_map = {occupation: index for index, occupation in enumerate(occupations)}

    #Import Data from u.user
    column_names = ["user_id", "age", "gender", "occupation", "zip_code"]
    users = pd.read_csv('u.user', sep='|', names=column_names)
    if verbose:
        print("Imported now hot encoding")

    #Hot Encode gender
    users['gender'] = users['gender'].apply(lambda x: 1 if x == 'F' else 0)

    # One-hot encode 'occupation'
    users['occupation'] = users['occupation'].apply(lambda x: occupation_map.get(x, -1))
    if verbose:
        print("Hot encoding done now Normalizing")
    # Normalize 'age'
    scaler = StandardScaler()
    users['age'] = scaler.fit_transform(users[['age']])

    # Drop the 'zip code' column
    users.drop(columns=['zip_code'], inplace=True)
    if verbose:
        print(users.columns)
    #Take out user_id to avoid clustering based on id
    users_data = users[['age', 'gender', 'occupation']]
    if verbose:
        print(users_data.columns)
        print("KMeans Clustering")
    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=3, random_state=15777)
    user_clusters = kmeans.fit_predict(users_data)

    # Add the cluster labels to the DataFrame
    users_data['cluster'] = user_clusters

    # Add the user_id labels to the DataFrame
    users_data.loc[:, 'user_id'] = users['user_id']

    return users_data


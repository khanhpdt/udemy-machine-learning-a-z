from data_preprocessing import data_preprocessing
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt

features = data_preprocessing.import_dataset('datasets/Mall_Customers.csv', [3, 4])

# choose K using elbow method
max_feature_count = 10
wcss = []
for i in range(1, max_feature_count + 1):
    kmeans = KMeans(n_clusters=i, init='k-means++', n_init=10, max_iter=300)
    kmeans.fit_predict(features)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, max_feature_count + 1), wcss)
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# look at the plot and choose the value using the elbow method
K = 5
kmeans = KMeans(n_clusters=K, init='k-means++', n_init=10, max_iter=300)
cluster_pred = kmeans.fit_predict(features)

# visualize the clusters. only applicable when the number of features is 2 or 3.
colors = ['red', 'blue', 'green', 'cyan', 'magenta']
for i in range(K):
    cluster_item_indexes = cluster_pred == i
    plt.scatter(features[cluster_item_indexes, 0], 
                features[cluster_item_indexes, 1], 
                c=colors[i],
                label='Cluster ' + str(i))
plt.scatter(kmeans.cluster_centers_[:, 0], 
            kmeans.cluster_centers_[:, 1], 
            s=300, 
            c='yellow',
            label='Centroids')
plt.legend()
plt.show()

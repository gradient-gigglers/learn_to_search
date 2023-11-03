import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
samples = np.array([[0, 0, 2], [1, 0, 0], [0, 0, 1]])
neigh = NearestNeighbors(n_neighbors=2, radius=0.4)
# neigh = NearestNeighbors(n_neighbors=2, radius=0.4, metric=cosine_similarity)
neigh.fit(samples)
result = neigh.kneighbors([[0, 0, 1.3]], 2, return_distance=False)
# result = neigh.kneighbors(np.array([[0, 0, 1.3]]), 2)
print(result)
import numpy as np
from sklearn.neighbors import NearestNeighbors
import numpy as np
from sklearn.decomposition import PCA
# Sample data - Replace this with your actual data
query_embedding = np.random.rand(128)  # Example query embedding
document_embeddings = np.random.rand(100, 128)  # Example document embeddings

# Define the number of nearest neighbors to retrieve
k = 5

# Apply PCA for dimensionality reduction
pca = PCA(n_components=50)  # Reduce from 128 to 50 dimensions, adjust as needed
document_embeddings_reduced = pca.fit_transform(document_embeddings)


# Perform KNN search
query_embedding_reduced = pca.transform(query_embedding.reshape(1, -1))  # Reduce the query embedding
nn_model = NearestNeighbors(n_neighbors=k, metric='euclidean', n_jobs=-1)  # Use 'euclidean' distance metric

# Fit the model with the document embeddings
nn_model.fit(document_embeddings_reduced)

# Perform KNN search for the query embedding
distances, indices = nn_model.kneighbors([query_embedding_reduced])

similar_document_embeddings = document_embeddings[indices[0]]

# Now, 'similar_document_embeddings' contains the 5 most similar document embeddings to the query embedding
document_text = doc_text_list[indices[0]]

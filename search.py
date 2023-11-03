import fastapi
import torch
import tokenizer
import model
import numpy as np
from sklearn.neighbors import NearestNeighbors
import numpy as np
from sklearn.decomposition import PCA
import SE2Updated as SE2


app = fastapi.FastAPI()


@app.on_event("startup")
async def startup_event():
  app.state.maybe_model = "Attach my model to some variables"
  app.state.tknz = (tokenizer.Tokenizer()).load_vocab("./vocab.txt")
  app.state.lang = model.Language(torch.rand(len(app.state.tknz.vocab), 50), 7)
  app.state.lang.load_state_dict(torch.load("./lang_epoch_5.pt"))
  app.state.langs = ["German", "Esperanto", "French", "Italian", "Spanish", "Turkish", "English"]
  app.state.lang.eval()


@app.on_event("shutdown")
async def startup_event():
  print("Shutting down")


@app.get("/")
def on_root():
  return { "message": "Hello App" }


@app.post("/what_language_is_this")
async def on_language_challenge(request: fastapi.Request):

  # The POST request body has a text filed,
  # take it and tokenize it. Then feed it to
  # the language model and return the result.
  text = (await request.json())["text"]
  tknz = app.state.tknz.encode(text)
  tknz = torch.tensor(tknz, dtype=torch.long).unsqueeze(0)
  if tknz.shape[1] == 0: return [
    {"class": class_name, "value": 1/len(app.state.langs)}
    for class_name in app.state.langs
  ]

  lang = app.state.lang(tknz)
  print("lang", lang)
  lang = torch.nn.functional.softmax(lang, dim=1)
  lang = lang.squeeze(0).tolist()
  result = [{"class": class_name, "value": value} for class_name, value in zip(app.state.langs, lang)]
  print("result", result)
  return result

# Sample data - Replace this with your actual data
query_model = SE2.BiLSTMWithPooling(query_tokens)
SE2.BiLSTMWithPooling.load_state_dict(torch.load("./query_epoch_5.pt"))


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

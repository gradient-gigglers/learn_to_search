import model
import tokenizer
import constants
import torch
import dask.dataframe as dd
import tokenizer as tknz
import torch.nn.functional as F

class QueryAndPassages:
    def __init__(self, query, passages):
        self.query = query
        self.passages = passages

class DataSetReader:
  def __init__(self, fileName=None, prefix='M'):
    self.tokenizer = tokenizer.Tokenizer(fileName,prefix)
    self.queriesAndPassages = []
    self.queries = []
    self.passages = []
    data = dd.read_parquet(fileName)
    if constants.PARQUET_NUM_OF_ROWS:
      data = data.head(constants.PARQUET_NUM_OF_ROWS, compute=True)
    self.model = model.CBOW(constants.VOCAB_SIZE, constants.DIMENSIONS)
    self.model.load_state_dict(torch.load(f"./cbow_epoch_{constants.NUM_OF_EMBEDDING_EPOCHS}.pt"))
    self.model.eval()
    
    # Iterate through DataFrame rows
    for index, item in data.iterrows():
      query = item['query']
      queryIndices = self.tokenizer.encode(query)
      queryEmbeddings = self.model.embeddings(torch.LongTensor(queryIndices))
      #queryEmbeddings = self.model.embeddings(torch.LongTensor(queryIndices)).detach()   Do we need to detach?
      passageEmbeddings = []
      for passage in item['passages']['passage_text']:
         passageIndices = self.tokenizer.encode(passage)
         passageEmbeddings.append(self.model.embeddings(torch.LongTensor(passageIndices)))
         #passageEmbedding = self.model.embeddings(torch.LongTensor(passageIndices)).detach()  Do we need to detach?

      # Determine the maximum length of passage embeddings for padding  
      max_length = max(len(passageEmbedding) for passageEmbedding in passageEmbeddings)

      # Pad each passageEmbedding tensor to the maximum length
      paddedPassageEmbeddings = []
      for passageEmbedding in passageEmbeddings:
        # Use F.pad for padding; pad on the sequence length dimension (dim=0)
        # 'pad' is a tuple indicating how many values to pad at the beginning and end of each dimension
        # Here we are padding with zeros on the left (0) and (max_length - len(passageEmbedding)) on the right
        padded = F.pad(passageEmbedding, pad=(0, 0, 0, max_length - len(passageEmbedding)), mode='constant', value=0)
        paddedPassageEmbeddings.append(padded)

      # Stack the padded passage embeddings. We are using stack to avoid creating a copy of teh tensors in a new tensor.
      passageEmbeddingsTensor = torch.stack(paddedPassageEmbeddings)

      self.queriesAndPassages.append(QueryAndPassages(queryEmbeddings, passageEmbeddingsTensor))
      self.queries.append([index, queryEmbeddings])
      self.passages.append([index, passageEmbeddingsTensor])
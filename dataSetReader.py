import model
import tokenizer
import constants
import torch
import dask.dataframe as dd
import tokenizer as tknz

class QueryAndPassages:
    def __init__(self, query, passages):
        self.query = query
        self.passages = passages

class DataSetReader:
  def __init__(self, fileName=None, prefix='M'):
    self.tokenizer = tokenizer.Tokenizer(fileName,prefix)
    self.queryAndPassages = []
    self.queries = []
    self.passages = []
    data = dd.read_parquet(fileName)
    if constants.PARQUET_NUM_OF_ROWS:
      data = data.head(constants.PARQUET_NUM_OF_ROWS, compute=True)
    self.model = model.CBOW(constants.VOCAB_SIZE, constants.DIMENSIONS)
    self.model.load_state_dict(torch.load(f"./cbow_epoch_{constants.EPOCHS}.pt"))
    self.model.eval()
    
    # Iterate through DataFrame rows
    for index, item in data.iterrows():
      query = item['query']
      queryIndices = self.tokenizer.encode(query)
      queryEmbeddings = self.model.embeddings(torch.LongTensor(queryIndices))
      passageEmbeddings = []
      for passage in item['passages']['passage_text']:
         passageIndices = self.tokenizer.encode(passage)
         passageEmbeddings.append(self.model.embeddings(torch.LongTensor(queryIndices)))
      self.queryAndPassages.append( QueryAndPassages(queryEmbeddings,passageEmbeddings))
      self.queries.append([index,queryEmbeddings] )
      self.passages.append([index,passageEmbeddings] )    
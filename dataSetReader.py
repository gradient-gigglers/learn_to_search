import pandas as pd
import model
import tokenizer
import train_cbow
import torch

class QueryAndPassages:
    def __init__(self, query, passages):
        self.query = query
        self.passages = passages

class DataSetReader:
  def __init__(self, fileName=None):
    self.queryAndPassages = []
    data = pd.read_parquet(fileName)
    self.model = model.CBOW(tokenizer.VOCAB_SIZE, train_cbow.DIMENSIONS)
    self.model.load_state_dict(torch.load(f"./cbow_epoch_{train_cbow.EPOCHS}.pt"))
    self.model.eval()
    
    # Iterate through DataFrame rows
    for index, item in data.iterrows():
      query = item['query']
      passages = item['passages']['passage_text']
      self.queryAndPassages.append( QueryAndPassages(query,passages) )    
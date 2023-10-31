import pandas as pd

class QueryAndPassages:
    def __init__(self, query, passages):
        self.query = query
        self.passages = passages

class DataSetReader:
  def __init__(self, fileName=None):
    self.queryAndPassages = []
    data = pd.read_parquet(fileName)
    # long_format = data.melt(value_vars=data.columns)

# Iterate through DataFrame rows
    for index, item in data.iterrows():
      query = item['query']
      passages = item['passages']['passage_text']
      # for index2, passage in item['passages'].iterrows():
      #    passage_text = passage['passage_text']
      #    passages.append(passage_text)
      self.queryAndPassages.append( QueryAndPassages(query,passages) )    
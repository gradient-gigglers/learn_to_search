import dataset as ds

if __name__ == '__main__':
  w2v_ds = ds.W2VData('../dataset/test/0000.parquet',"passage",3)
  print("word2vec:corpus[0]", w2v_ds.tokenizer.corpus[:10] )
  print("word2vec:ds[0]", w2v_ds.data[0])
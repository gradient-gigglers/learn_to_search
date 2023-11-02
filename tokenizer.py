import string
import sentencepiece as spm
from pandas import json_normalize
import json
from pathlib import Path
import constants 
import sys
import dask.dataframe as dd

class Tokenizer:
  def __init__(self, fileName=None, prefix='M'):
    filePath = Path(prefix+'.model')
    if not filePath.exists():
      data = dd.read_parquet(fileName)
      if constants.PARQUET_NUM_OF_ROWS:
        data = data.head(constants.PARQUET_NUM_OF_ROWS, compute=True)
      passage_texts = [text for passage in data['passages'] for text in passage['passage_text']]
      if sys.platform.startswith('win'):
        with open(prefix+'_corpus.txt', 'w', encoding="UTF-8") as f:
          for passage in passage_texts: f.write(passage + '\n')
      else:
        with open(prefix+'_corpus.txt', 'w') as f:
          for passage in passage_texts: f.write(passage + '\n')
      # Train the SentencePiece model
      spm.SentencePieceTrainer.train('--input=' + prefix + '_corpus.txt --model_prefix=' + prefix + ' --vocab_size=' + str(constants.VOCAB_SIZE))
    self.sp = spm.SentencePieceProcessor(model_file=prefix+'.model')
    with open(prefix + '_corpus.txt', 'r', encoding="UTF-8") as f: self.corpus = [line.strip() for line in f.readlines()]
    self.vocab = [self.sp.IdToPiece(id) for id in range(self.sp.GetPieceSize())] 
    
    # self.word2idx = {word: idx for idx, word in enumerate(self.vocab)}
    # self.idx2word = {idx: word for idx, word in enumerate(self.vocab)}

  def encode(self, sentence):
    return self.sp.encode_as_ids(sentence)
    
  def decode(self, indices):
    return self.sp.decode(indices)
  
  def get_corpus(self):
    return self.corpus

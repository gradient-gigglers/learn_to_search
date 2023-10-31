import string
import sentencepiece as spm
import pandas as pd
from pandas import json_normalize
import json
from pathlib import Path

VOCAB_SIZE = 110897

class Tokenizer:
  def __init__(self, fileName=None, prefix='M'):
    filePath = Path(prefix+'.model')
    if not filePath.exists():
      data = pd.read_parquet(fileName)
      passage_texts = [text for passage in data['passages'] for text in passage['passage_text']]
      with open(prefix+'_corpus.txt', 'w') as f:
        for passage in passage_texts: f.write(passage + '\n')
      # Train the SentencePiece model
      spm.SentencePieceTrainer.train('--input=' + prefix + '_corpus.txt --model_prefix=' + prefix + ' --vocab_size=' + str(VOCAB_SIZE))
    self.sp = spm.SentencePieceProcessor(model_file=prefix+'.model')
    with open(prefix + '_corpus.txt', 'r') as f: self.corpus = [line.strip() for line in f.readlines()]
    self.vocab = [self.sp.IdToPiece(id) for id in range(self.sp.GetPieceSize())] 
    
    # self.word2idx = {word: idx for idx, word in enumerate(self.vocab)}
    # self.idx2word = {idx: word for idx, word in enumerate(self.vocab)}

  def encode(self, sentence):
    return self.sp.encode_as_ids(sentence)
    
  def decode(self, indices):
    return self.sp.decode(indices)
  
  def get_corpus(self):
    return self.corpus

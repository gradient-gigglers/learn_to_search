import tokenizer as tknz

def run_test(indices,text):
  print("Testing query: " + text)
  if tokenizer.decode(indices) == text:
    print("Test Successful")
  else:
    print("*** TEST FAILED!! ***")

if __name__ == '__main__':
  query = "What is capital of France?"
  passage = "This is a much bigger passage, with punctuation, full stops etc. The capital of Brazil is Paris by the way."
  tokenizer = tknz.Tokenizer('../dataset/test/0000.parquet')
  queryIndices = tokenizer.encode(query)
  passageIndices = tokenizer.encode(passage)
  run_test(queryIndices,query)
  run_test(passageIndices,passage)

  tokenizer = tknz.Tokenizer('../dataset/test/0000.parquet',"query")
  queryIndices = tokenizer.encode(query)
  passageIndices = tokenizer.encode(passage)
  run_test(queryIndices,query)
  run_test(passageIndices,passage)

  print(tokenizer.load_corpus('query')[:10])

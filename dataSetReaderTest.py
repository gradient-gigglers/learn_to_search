import dataSetReader as dsr

if __name__ == '__main__':
  datasetReader = dsr.DataSetReader('../dataset/test/0000.parquet','passsage')   
  print(len(datasetReader.queryAndPassages))
  for i in range(10):
    print(datasetReader.queryAndPassages[i].query)
    numPassages = len(datasetReader.queryAndPassages[i].passages)
    for j in range(numPassages):
      print(datasetReader.queryAndPassages[i].passages[j])

  print('---------------')
  print('Indexed Queries')
  print('---------------')
  for i in range(10):
    print(datasetReader.queries[i])

  print('----------------')
  print('Indexed Passages')
  print('----------------')
  for i in range(10):
    print(datasetReader.passages[i])
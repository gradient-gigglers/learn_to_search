import dataSetReader as dsr

if __name__ == '__main__':
  datasetReader = dsr.DataSetReader('../dataset/test/0000.parquet')   
  print(len(datasetReader.queryAndPassages))
  for i in range(5):
    print(datasetReader.queryAndPassages[i].query)
    numPassages = len(datasetReader.queryAndPassages[0].passages)
    for j in range(numPassages):
      print(datasetReader.queryAndPassages[i].passages[j-1])
  

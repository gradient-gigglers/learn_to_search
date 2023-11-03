import dataSetReader as dsr
import constants

if __name__ == '__main__':
  datasetReader = dsr.DataSetReader('../dataset/' + constants.DATASET + '/0000.parquet','passsage')   
  print(len(datasetReader.queriesAndPassages))
  for i in range(10):
    print(datasetReader.queriesAndPassages[i].query)
    numPassages = len(datasetReader.queriesAndPassages[i].passages)
    for j in range(numPassages):
      print(datasetReader.queriesAndPassages[i].passages[j])

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
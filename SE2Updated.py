import torch
import torch.nn as nn
import torch.optim as optim
from random import choice, randint
import numpy as np
import dataSetReader as dsr
import constants

class BiLSTMWithPooling(nn.Module):
    def __init__(self, input_size, hidden_dim, output_size,num_layers):
        super(BiLSTMWithPooling, self).__init__()

        self.bilstm = nn.LSTM(input_size, hidden_dim,output_size, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(2 * hidden_dim,output_size)  # 2 * hidden_dim due to bidirectional

    def forward(self, x):
        x=x[0]
        out, _ = self.bilstm(x)
        fc_input = out[-1]

        # Fully connected layer
        output = self.fc(fc_input)
        return output

# Define model hyperparameters
hidden_dim = 64
input_size = 1000
num_layers = 1  # You can adjust the number of layers 
output_size = 200


############## SIAMESE MODEL ################
# Create two instances of the BiLSTM model (shared weights)
query_model = BiLSTMWithPooling(input_size, hidden_dim, output_size,num_layers,)
doc_model = BiLSTMWithPooling(input_size, hidden_dim, output_size, num_layers)

# Ensure weight sharing by setting model_right's weights to be the same as model_left's
query_model.load_state_dict(doc_model.state_dict())

loss_fn_triplet = nn.TripletMarginLoss(margin=0.2, p=2)
queriesAndPassages = dsr.DataSetReader('../dataset/' + constants.DATASET + '/0000.parquet','passsage').queriesAndPassages

#############TRAINING CYCLE################

num_epochs = 2
for epoch in range(num_epochs): 

    #select positive and negative document samples for training given specific query
    for queryAndPassages in queriesAndPassages:

      #fill anchor tensor with query embedding
      
      anchors = queryAndPassages.query
      positives = queryAndPassages.passages

      #find negative examples
      while True:
        rand_index = randint(0, len(queriesAndPassages)-1)
        #if queriesAndPassages[rand_index].query != anchors:
        break
      
      #choice(i for i in range(0,len(query_list)))
      negatives = queriesAndPassages[rand_index].passages
 
      #calculate input size for anchor
      embed_anchors = query_model(anchors)
      embed_positives = doc_model(positives)
      embed_negatives = doc_model(negatives)

      #calculate triplet loss for batch of samples
      """"
      def cosine_similarity(x1,x2):
          out = F.cosine_similarity(x1,x2, dim=1)
          return out
      """
      #loss_fn_triplet = nn.TripletMarginWithDistanceLoss(distance_function=cosine_similarity,margin=0.4) 
      triplet_loss = loss_fn_triplet(embed_anchors,embed_positives,embed_negatives)

      #Perform backpropagation and optimization
      optimizer = optim.Adam(query_model.parameters(), lr=0.01)
      optimizer.zero_grad()
      triplet_loss.backward()
      optimizer.step()

      # Print the triplet loss to monitor training progress
      print("Triplet Loss:", triplet_loss.item())
      print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {triplet_loss.item()}')

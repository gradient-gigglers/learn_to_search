import torch
import torch.nn as nn
import torch.optim as optim
from random import choice, randint
import numpy as np
import dataSetReader as dsr
import constants
import torch.nn.functional as F

class BiLSTMWithPooling(nn.Module):
    def __init__(self, input_size, hidden_dim, output_size,num_layers):
        super(BiLSTMWithPooling, self).__init__()

        self.bilstm = nn.LSTM(input_size, hidden_dim,output_size, batch_first=True, bidirectional=True, num_layers=num_layers)
        self.fc = nn.Linear(2 * hidden_dim,output_size)  # 2 * hidden_dim due to bidirectional

    def forward(self, x):
        #x=x[0]
        out, _ = self.bilstm(x)
        if out.ndim == 3:
            fc_input = out[:,-1,:]
        else:
            fc_input = out[-1]

        # Fully connected layer
        output = self.fc(fc_input)
        return output

# Define model hyperparameters
hidden_dim = constants.LSTM_HIDDEN_DIM
input_size = constants.LSTM_INPUT_SIZE
num_layers = constants.LSTM_NUM_LAYERS  # You can adjust the number of layers 
output_size = constants.MODEL_OUTPUT_SIZE


############## SIAMESE MODEL ################
# Create two instances of the BiLSTM model (shared weights)
query_model = BiLSTMWithPooling(input_size, hidden_dim, output_size,num_layers,)
doc_model = BiLSTMWithPooling(input_size, hidden_dim, output_size, num_layers)

# Ensure weight sharing by setting model_right's weights to be the same as model_left's
query_model.load_state_dict(doc_model.state_dict())

loss_fn_triplet = nn.TripletMarginLoss(margin=constants.TRIPLE_LOSS_MARGIN, p=constants.TRIPLE_LOSS_PAIRWISE_DISTANCE)
queriesAndPassages = dsr.DataSetReader('../dataset/' + constants.DATASET + '/0000.parquet','passsage').queriesAndPassages

#############TRAINING CYCLE################

num_epochs = constants.NUM_OF_MODEL_EPOCHS
for epoch in range(num_epochs): 

    #select positive and negative document samples for training given specific query
    for queryAndPassages in queriesAndPassages[:constants.NUM_OF_QUERIES_FOR_MODEL]:

      #fill anchor tensor with query embedding
      
      anchors = queryAndPassages.query
      positives = queryAndPassages.passages

      #find negative examples
      negatives = []
 
      seq_len_positives = len(positives)  
      while len(negatives) < seq_len_positives:
        rand_index = randint(0, len(queriesAndPassages)-1)
        #if queriesAndPassages[rand_index].query != anchors:
        negatives.append(queriesAndPassages[rand_index].passages[0])

      # Determine the maximum length of passage embeddings for padding  
      max_length = max(len(negative) for negative in negatives)

      # Pad each passage to the maximum length
      paddedNegativeEmbeddings = []
      for negative in negatives:
        # Use F.pad for padding; pad on the sequence length dimension (dim=0)
        # 'pad' is a tuple indicating how many values to pad at the beginning and end of each dimension
        # Here we are padding with zeros on the left (0) and (max_length - len(passageEmbedding)) on the right
        padded = F.pad(negative, pad=(0, 0, 0, max_length - len(negative)), mode='constant', value=0)
        paddedNegativeEmbeddings.append(padded)


      #calculate input size for anchor
      embed_anchors = query_model(anchors)
      embed_positives = doc_model(positives)
      embed_negatives = doc_model(torch.stack(paddedNegativeEmbeddings))

      #calculate triplet loss for batch of samples
      """"
      def cosine_similarity(x1,x2):
          out = F.cosine_similarity(x1,x2, dim=1)
          return out
      """
      #loss_fn_triplet = nn.TripletMarginWithDistanceLoss(distance_function=cosine_similarity,margin=0.4) 
      embed_anchors_list = []
      embed_anchors_list.append(embed_anchors)
      triplet_loss = loss_fn_triplet(torch.stack(embed_anchors_list),embed_positives,embed_negatives)

      #Perform backpropagation and optimization
      optimizer = optim.Adam(query_model.parameters(), lr=constants.MODEL_LERNING_RATE)
      optimizer.zero_grad()
      triplet_loss.backward(retain_graph=True)
      optimizer.step()

      # Print the triplet loss to monitor training progress
      print("Triplet Loss:", triplet_loss.item())
      print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {triplet_loss.item()}')

    torch.save(query_model.state_dict(), f"./query_epoch_{epoch+1}.pt")
    torch.save(doc_model.state_dict(), f"./doc_epoch_{epoch+1}.pt")  # These are probably same but just to check

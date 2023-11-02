import torch
import dataset as ds
import pandas
import model
import tqdm
import wandb
import tokenizer
import constants

# start a new wandb run to track this script
if constants.WANDB_ON:
  wandb.init(
      # set the wandb project where this run will be logged
      project="test-project",
      
      # track hyperparameters and run metadata
      config={
      "learning_rate": constants.LEARNING_RATE,
      "dimensions": constants.DIMENSIONS,
      "dataset": constants.DATASET,
      "vocab_size": constants.VOCAB_SIZE,
      "epochs": constants.EPOCHS,
      }
  )

w2v_ds = ds.W2VData(F"../dataset/{constants.DATASET}/0000.parquet","passage",constants.CBOW_WINDOW_SIZE)
dl = torch.utils.data.DataLoader(w2v_ds, batch_size=constants.BATCH_SIZE, shuffle=True)

cbow = model.CBOW(constants.VOCAB_SIZE, constants.DIMENSIONS)
loss_function = torch.nn.NLLLoss()
optimizer = torch.optim.SGD(cbow.parameters(), lr=constants.LEARNING_RATE)

for epoch in range(constants.EPOCHS):
  total_loss = 0
  for context, target in tqdm.tqdm(dl, desc=f"Epoch {epoch+1}/{constants.EPOCHS}", unit="batch"):
    optimizer.zero_grad()
    log_probs = cbow(context)
    loss = loss_function(log_probs, target)
    loss.backward()
    optimizer.step()
    total_loss += loss.item()
  torch.save(cbow.state_dict(), f"./cbow_epoch_{epoch+1}.pt")  
  if constants.WANDB_ON:
    wandb.log({"acc": 2, "total_loss": total_loss})
  
if constants.WANDB_ON:
  wandb.finish()

  
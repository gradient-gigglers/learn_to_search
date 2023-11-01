import torch
import dataset as ds
import pandas
import model
import tqdm
import wandb
import tokenizer

LEARNING_RATE = 0.02
DIMENSIONS = 20
DATASET = "test"
VOCAB_SIZE = tokenizer.VOCAB_SIZE
EPOCHS = 2

# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="test-project",
    
    # track hyperparameters and run metadata
    config={
    "learning_rate": LEARNING_RATE,
    "dimensions": DIMENSIONS,
    "dataset": DATASET,
    "vocab_size": VOCAB_SIZE,
    "epochs": EPOCHS,
    }
)

w2v_ds = ds.W2VData(F"../dataset/{DATASET}/0000.parquet","passage",3)
dl = torch.utils.data.DataLoader(w2v_ds, batch_size=4, shuffle=True)

cbow = model.CBOW(VOCAB_SIZE, DIMENSIONS)
loss_function = torch.nn.NLLLoss()
optimizer = torch.optim.SGD(cbow.parameters(), lr=0.001)

for epoch in range(EPOCHS):
  total_loss = 0
  for context, target in tqdm.tqdm(dl, desc=f"Epoch {epoch+1}/{EPOCHS}", unit="batch"):
    optimizer.zero_grad()
    log_probs = cbow(context)
    loss = loss_function(log_probs, target)
    loss.backward()
    optimizer.step()
    total_loss += loss.item()
  torch.save(cbow.state_dict(), f"./cbow_epoch_{epoch+1}.pt")  
  wandb.log({"acc": 2, "total_loss": total_loss})
  
wandb.finish()

  
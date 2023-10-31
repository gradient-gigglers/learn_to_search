import torch
import dataset as ds
import pandas
import model
import tqdm

w2v_ds = ds.W2VData('../dataset/test/0000.parquet',"passage",3)
dl = torch.utils.data.DataLoader(w2v_ds, batch_size=4, shuffle=True)

cbow = model.CBOW(len(w2v_ds.tokenizer.vocab), 50)
loss_function = torch.nn.NLLLoss()
optimizer = torch.optim.SGD(cbow.parameters(), lr=0.001)


for epoch in range(5):
  total_loss = 0
  for context, target in tqdm.tqdm(dl, desc=f"Epoch {epoch+1}/10", unit="batch"):
    optimizer.zero_grad()
    log_probs = cbow(context)
    loss = loss_function(log_probs, target)
    loss.backward()
    optimizer.step()
    total_loss += loss.item()
  print(f"Epoch {epoch+1}/10, Loss: {total_loss}")
  torch.save(cbow.state_dict(), f"./cbow_epoch_{epoch+1}.pt")
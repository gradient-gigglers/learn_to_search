import torch

class CBOW(torch.nn.Module):
  def __init__(self, vocab_size, embedding_dim):
    super(CBOW, self).__init__()
    self.embeddings = torch.nn.Embedding(vocab_size, embedding_dim)
    self.linear = torch.nn.Linear(embedding_dim, vocab_size)

  def forward(self, inputs):
    embeds = torch.sum(self.embeddings(inputs), dim=1)
    out = self.linear(embeds)
    log_probs = torch.nn.functional.log_softmax(out, dim=1)
    return log_probs

if __name__ == '__main__':
  cbow = CBOW(20000, 50)
  emb_weights = cbow.embeddings.weight.data # shape(20.000, 50)
  sentence = torch.tensor([[5, 89, 3]]) # shape(1, 3)
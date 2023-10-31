import torch
import tokenizer


class W2VData(torch.utils.data.Dataset):
  def __init__(self, fileName, prefix, window_size=2):
    self.tokenizer = tokenizer.Tokenizer(fileName,prefix)
    self.data = []
    self.create_data(window_size)

  def create_data(self, window_size):
    for sentence in self.tokenizer.corpus:
      tokens = self.tokenizer.encode(sentence)
      for i, target in enumerate(tokens):
        context = tokens[max(0, i - window_size):i] + tokens[i + 1:i + window_size + 1]
        if len(context) != 2 * window_size: continue
        self.data.append((context, target))

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    context, target = self.data[idx]
    return torch.tensor(context), torch.tensor(target)

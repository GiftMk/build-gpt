import torch

def encode_dataset(dataset, tokenizer):
  data = torch.tensor(tokenizer.encode(dataset), dtype=torch.long)
  n = int(0.9 * len(data))
  training_data = data[:n]
  validation_data = data[n:]

  return training_data, validation_data
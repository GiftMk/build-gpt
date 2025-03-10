import torch
from get_batch import get_batch

BATCH_SIZE = 32

def train(data, model, num_steps):
  optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
  loss = None

  for _ in range(num_steps):
    inputs, targets = get_batch(data, BATCH_SIZE)
    _, loss = model(inputs, targets)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    loss = loss.item()

  return loss


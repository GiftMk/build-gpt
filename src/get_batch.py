import torch
from constants import SEED, DEVICE

torch.manual_seed(SEED)

# TODO: document what the purpose of function
def get_batch(data, batch_size, block_size):
  random_indices = torch.randint(len(data) - block_size, (batch_size,))
  inputs = get_inputs(data, random_indices, block_size)
  targets = get_targets(data, random_indices, block_size)

  return inputs.to(DEVICE), targets.to(DEVICE)

def get_inputs(data, indices, block_size):
  inputs = []

  for i in indices:
    input_array = data[i:i + block_size]
    inputs.append(input_array)

  return torch.stack(inputs)

def get_targets(data, indices, block_size):
  targets = []

  for i in indices:
    target_array = data[i + 1: i + block_size + 1]
    targets.append(target_array)

  return torch.stack(targets)
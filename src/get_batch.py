import torch

torch.manual_seed(1337)

BATCH_SIZE = 4
BLOCK_SIZE = 8

def get_batch(data):
  random_indices = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
  inputs = get_inputs(data, random_indices)
  targets = get_targets(data, random_indices)

  return inputs, targets

def get_inputs(data, indices):
  inputs = []

  for i in indices:
    input_array = data[i:i + BLOCK_SIZE]
    inputs.append(input_array)

  return torch.stack(inputs)

def get_targets(data, indices):
  targets = []

  for i in indices:
    target_array = data[i + 1: i + BLOCK_SIZE + 1]
    targets.append(target_array)

  return torch.stack(targets)
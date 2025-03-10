import torch

torch.manual_seed(1337)

# TODO: document what these values mean and why we chose them
# TODO: explore performance when using different values
BLOCK_SIZE = 8

# TODO: document what the purpose of function
def get_batch(data, batch_size):
  random_indices = torch.randint(len(data) - BLOCK_SIZE, (batch_size,))
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
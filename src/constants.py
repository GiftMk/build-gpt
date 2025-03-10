import torch

SEED=1337
DEVICE= 'cuda' if torch.cuda.is_available() else 'cpu'
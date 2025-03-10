import torch
import torch.nn as nn
from torch.nn.functional import cross_entropy, softmax

torch.manual_seed(1337)

# TODO: Watch the make more series to understand this class. Then add documentation here
class BigramLanguageModel(nn.Module):
  def __init__(self, vocab_size):
    super().__init__()
    self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

  def forward(self, inputs, targets=None):
    logits = self.token_embedding_table(inputs)

    if (targets is None):
      return logits, None
    
    B, T, C = logits.shape
    logits = logits.view(B * T, C)
    targets = targets.view(B * T)

    loss = cross_entropy(logits, targets)

    return logits, loss
  
  def generate(self, inputs, max_new_tokens):
    for _ in range(max_new_tokens):
      logits, _ = self(inputs)
      probabilities = softmax(logits[:, -1, :], dim=1)
      prediction = torch.multinomial(probabilities, num_samples=1)
      inputs = torch.cat((inputs, prediction), dim=1)
    return inputs
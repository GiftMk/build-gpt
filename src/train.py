import torch
from get_batch import get_batch
from collections import namedtuple
from estimate_loss import estimate_loss

TrainingSettings = namedtuple('TrainingSettings', ['batch_size', 'block_size', 'num_steps', 'evaluation_interval'])

def train(training_data, validation_data, model, settings):
  optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

  for i in range(settings.num_steps):
    if (should_evaluate(i, settings.evaluation_interval)):
      training_loss, validation_loss = estimate_loss(training_data, validation_data, model, settings)
      print(f"step {i}: training loss {training_loss:.4f}, validation loss {validation_loss:.4f}")

    inputs, targets = get_batch(data=training_data, batch_size=settings.batch_size, block_size=settings.block_size)
    _, loss = model(inputs, targets)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

def should_evaluate(training_step, evaluation_interval):
  return training_step % evaluation_interval == 0
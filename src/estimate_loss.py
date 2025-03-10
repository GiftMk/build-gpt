import torch
from get_batch import get_batch

def estimate_loss(training_data, validation_data, model, settings):
  training_loss = None
  validation_loss = None

  #TODO: document what eval and train mode are for
  model.eval()

  with torch.no_grad():
    for mode in ['training', 'validation']:
      losses = torch.zeros(settings.num_steps)
      data = training_data if mode == 'training' else validation_data
      inputs, targets = get_batch(data, batch_size=settings.batch_size, block_size=settings.block_size)

      for i in range(settings.num_steps):
        _, loss = model(inputs, targets)
        losses[i] = loss.item()
      
      if mode == 'training':
        training_loss = losses.mean()
      else:
        validation_loss = losses.mean()
  
  model.train()

  return training_loss, validation_loss

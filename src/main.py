from load_dataset import load_dataset 
from tokenizer import Tokenizer
from encode_dataset import encode_dataset
from model import BigramLanguageModel
from train import train
import torch
from constants import DEVICE
from train import TrainingSettings

dataset, vocab = load_dataset()
vocab_size = len(vocab)

tokenizer = Tokenizer(vocab)
training_data, validation_data = encode_dataset(dataset, tokenizer)

model = BigramLanguageModel(vocab_size).to(DEVICE)
# TODO: document what these values mean and why we chose them
# TODO: explore performance when using different values
training_settings = TrainingSettings(batch_size=32, block_size=8, num_steps=50_000, evaluation_interval=300)
train(training_data, validation_data, model, training_settings)

test_input = torch.zeros((1, 1), dtype=torch.long)
prediction = model.generate(test_input, max_new_tokens=100)[0].tolist()

print(tokenizer.decode(prediction))

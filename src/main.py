from load_dataset import load_dataset 
from tokenizer import Tokenizer
from encode_dataset import encode_dataset
from language_model import BigramLanguageModel
from train import train
import torch

dataset, vocab = load_dataset()
vocab_size = len(vocab)

tokenizer = Tokenizer(vocab)
training_data, validation_data = encode_dataset(dataset, tokenizer)

model = BigramLanguageModel(vocab_size)
loss = train(training_data, model, num_steps=100_000)

test_input = torch.zeros((1, 1), dtype=torch.long)
prediction = model.generate(test_input, max_new_tokens=100)[0].tolist()

print(loss, tokenizer.decode(prediction))

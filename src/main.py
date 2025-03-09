from load_dataset import load_dataset 
from tokenizer import Tokenizer
from encode_dataset import encode_dataset
from get_batch import get_batch

dataset = load_dataset()
distinct_characters = sorted(list(set(dataset)))
tokenizer = Tokenizer(distinct_characters)
training_data, validation_data = encode_dataset(dataset, tokenizer)
inputs, targets = get_batch(training_data)

print('inputs', inputs)
print('targets', targets)
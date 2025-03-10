from urllib.request import urlopen
from os import path

DATASET_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
OUTPUT_FILENAME = 'training-data.txt'
OUTPUT_FILEPATH = path.join('..', OUTPUT_FILENAME)

def load_dataset():
  dataset = None
  try:
    with open(OUTPUT_FILEPATH, "r") as file:
      dataset = file.read()
  except:
    with urlopen(DATASET_URL) as response:
      dataset = response.read().decode('utf-8')
      save_dataset(dataset)
      dataset = dataset
    
  if dataset is None:
    raise Exception("Failed to load dataset")
  
  vocab = sorted(list(set(dataset)))

  return dataset, vocab

def save_dataset(dataset):
  with open(OUTPUT_FILEPATH, "w") as file:
    file.write(dataset)
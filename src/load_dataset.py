from urllib.request import urlopen
from os import path

DATASET_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
OUTPUT_FILENAME = 'training-data.txt'
OUTPUT_FILEPATH = path.join('..', OUTPUT_FILENAME)

def load_dataset():
  try:
    with open(OUTPUT_FILEPATH, "r") as file:
      return file.read()
  except:
    with urlopen(DATASET_URL) as response:
      dataset = response.read().decode('utf-8')
      save_dataset(dataset)
      return dataset

def save_dataset(dataset):
  with open(OUTPUT_FILEPATH, "w") as file:
    file.write(dataset)
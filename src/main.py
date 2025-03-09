from load_dataset import load_dataset 
from tokenizer import Tokenizer

dataset = load_dataset()
tokenizer = Tokenizer(dataset)

encoded_text = tokenizer.encode("Hi Mum, I love you")
print("Encoded", encoded_text)

decoded_text = tokenizer.decode(encoded_text)
print("Decoded", decoded_text)
# BUILD GPT

This is a follow along to Andrej Karpathy's [awesome video](https://www.youtube.com/watch?v=kCc8FmEb1nY&t=6533s)on building a tiny version of OpenAI's Chat-GPT!

# Prerequisites

- You have python3 installed on your machine.
- This repo uses the standard `urllib` library to fetch the training dataset. To successfully make
  GET requests, I first needed to run the `Install Certificates.command` script which on my Macbook
  was located in my `Applications/Python 3.11` folder.

# Architecture

## Training Data

For our training data, we are using Andrej's Tiny Shakespeare dataset. With is an open-source aggregation of all (or at least a large volume) of Shakespeare's written work.

## Tokenizer

We implement our own character-level tokenizer.
Basically, we find all the distinct characters in our dataset and assign each
a unique integer value based on it's index.

This allows us to encode an arbitrary string into tokens (unique integers). And decode
an array of tokens into a string.

TODO:

- [ ] Explore what happens when the dataset doesn't contain the complete alphabet.
- [ ] Gain a better understanding on the different tokenization approaches:
  - character
  - word
  - sub-word

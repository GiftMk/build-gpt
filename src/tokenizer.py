class Tokenizer:
  def __init__(self, dataset):
    distinct_characters = sorted(list(set(dataset)))
    self.__char_to_int_dict = { char:i for i, char in enumerate(distinct_characters) }
    self.__int_to_char_dict = { i:char for i, char in enumerate(distinct_characters) }

  def encode(self, value):
    return [self.__char_to_int_dict[char] for char in value]

  def decode(self, value):
    return ''.join([self.__int_to_char_dict[i] for i in value])
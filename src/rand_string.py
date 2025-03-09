import random
import string

LETTERS = list(string.ascii_letters)

def rand_string(length):
  if (length <= 0):
    return ''

  result = ''

  for _ in range(length):
    i = random.randint(0, len(LETTERS) - 1)
    letter = LETTERS[i]
    result += letter

  return result

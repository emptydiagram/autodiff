import matplotlib.pyplot as plt
import torch

DEFAULT_FILE_NAME = 'names.txt'
CHARS = '.abcdefghijklmnopqrstuvwxyz'
CHAR_INDICES = { c:i for (i, c) in enumerate(CHARS) }

def load_words_from_file(filename=None):
  if filename is None:
    filename = DEFAULT_FILE_NAME
  with open(filename) as f:
    words = f.read().splitlines()
  return words

def bigram_model(words=None):
  if words is None:
    words = load_words_from_file()

  # count up bigrams
  counts = torch.zeros((len(CHARS), len(CHARS)), dtype=torch.int32)

  for word in words:
    word = f".{word}."
    for i in range(len(word) - 1):
      idx_a = CHAR_INDICES[word[i]]
      idx_b = CHAR_INDICES[word[i+1]]
      counts[idx_a, idx_b] += 1
  
  return counts





if __name__ == '__main__':
  bigram_model()
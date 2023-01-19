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


def plot_bigram_frequencies(counts):
  plt.figure(figsize=(16,16))
  plt.imshow(counts, cmap='Greens')
  m = len(CHARS)
  for i in range(m):
    for j in range(m):
      plt.text(j, i, f"{CHARS[i]}{CHARS[j]}", ha="center", va="bottom", color="gray")
      plt.text(j, i, counts[i,j].item(), ha="center", va="top", color="gray")

def calculate_anll_loss(words, count_dists):
  sum_ll = 0.
  n = 0
  for word in words:
    word = f".{word}."
    for i in range(len(word)-1):
      idx_a = CHAR_INDICES[word[i]]
      idx_b = CHAR_INDICES[word[i+1]]
      sum_ll += torch.log(count_dists[idx_a, idx_b])
      n += 1
  anll = -sum_ll/ float(n)
  return anll

def make_count_dists(counts):
  count_dists = (counts+1).float()
  count_dists /= count_dists.sum(1, keepdim=True)
  return count_dists


if __name__ == '__main__':
  bigram_model()
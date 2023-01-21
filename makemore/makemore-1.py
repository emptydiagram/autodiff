import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

DEFAULT_FILE_NAME = 'names.txt'
CHARS = '.abcdefghijklmnopqrstuvwxyz'
CHAR_INDICES = { c:i for (i, c) in enumerate(CHARS) }

def load_words_from_file(filename=None):
  if filename is None:
    filename = DEFAULT_FILE_NAME
  with open(filename) as f:
    words = f.read().splitlines()
  return words


### Bigram count methods

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


### Neural net methods

def make_bigram_char_dataset(words):
  xs = []
  ys = []

  for word in words:
    word = f".{word}."
    for i in range(len(word) - 1):
      idx_a = CHAR_INDICES[word[i]]
      idx_b = CHAR_INDICES[word[i+1]]
      xs.append(idx_a)
      ys.append(idx_b)
  xs = torch.tensor(xs)
  ys = torch.tensor(ys)
  return (xs, ys)

def run_gradient_descent(W, X, Y, gd_params=None):
  if gd_params is None:
    # gd_params = {}
    gd_params = {
      'learning_rate' : 50.0,
      'num_iters': 100,
      'reg_param': 0.005
    }

  learning_rate = 50.0 if 'learning_rate' not in gd_params else gd_params['learning_rate']
  num_iters = 100 if 'num_iterations' not in gd_params else gd_params['num_iterations']
  reg_param = 0.005 if 'regularization_coeff' not in gd_params else gd_params['regularization_coeff']

  print(f"Running gradient descent with params: {{ {learning_rate=}, {num_iters=}, {reg_param=} }}")

  for i in range(num_iters):
    # forward pass
    logits = X @ W
    sm_counts = logits.exp()
    probs = sm_counts / sm_counts.sum(1, keepdims=True)

    # loss with weight decay
    loss = -probs[torch.arange(Y.shape[0]), Y].log().mean() + reg_param * (W**2).mean()

    if i % 10 == 0:
      print(f"iter #{i}, loss = {loss.item()}")

    W.grad = None
    loss.backward()
    W.data += -learning_rate * W.grad

  # forward pass
  logits = X @ W
  sm_counts = logits.exp()
  probs = sm_counts / sm_counts.sum(1, keepdims=True)

  # loss with weight decay
  loss = -probs[torch.arange(Y.shape[0]), Y].log().mean() + reg_param * (W**2).mean()
  print(f"iter #{num_iters}, loss = {loss.item()}")


def make_logreg_net(gen):
  # TODO: use GPU if available?
  #print(f"cuda is available? {torch.cuda.is_available()}")
  device = torch.device("cpu")
  W = torch.randn((27, 27), generator=gen, requires_grad=True, device=device)
  print(f"Made linear layer with weights shape = {W.shape}")
  return W

def logreg_net_preprocess_data(xs, Y):
  X = F.one_hot(xs, num_classes=27).float()
  print(f"Made data with (X, Y) shape = ({X.shape}, {Y.shape})")
  return (X, Y)


if __name__ == '__main__':
  bigram_model()
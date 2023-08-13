import numpy as np
from model import model



## Data load and preprocessing

data = open('src/data/dinos.txt', 'r').read()
data= data.lower()
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
print('There are %d total characters and %d unique characters in your data.' % (data_size, vocab_size))

chars = sorted(chars) ## sorting characters
char_to_ix = { ch:i for i,ch in enumerate(chars) } ## char to integer encoding
ix_to_char = { i:ch for i,ch in enumerate(chars) } ## integer to char encoding


parameters, last_name = model(data.split("\n"), ix_to_char, char_to_ix, 22001, verbose = True)


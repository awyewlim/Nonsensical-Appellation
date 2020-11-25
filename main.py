import numpy as np
import random
from model import *

data = open('training.txt', 'r').read()
data= data.lower()
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
#print('There are %d total characters and %d unique characters in your data.' % (data_size, vocab_size))

chars = sorted(chars)
#print(chars)

char_to_ix = { ch:i for i,ch in enumerate(chars) }
ix_to_char = { i:ch for i,ch in enumerate(chars) }

parameters = model(data, ix_to_char, char_to_ix, num_iterations = 100000, names = 0, vocab_size = vocab_size, verbose = True)

print('Welcome to Gaming Name Generator!' + '\n')
more = None
while more != 'N':
  num = int(input('Numbers of names you are looking for: '))
  print('\n')
  for i in range(num):
    sampled_indices = sample(parameters, char_to_ix)
    print_sample(sampled_indices, ix_to_char)
  print('\n')
  more = input('Looking more? (Enter "N" to exit): ')
print('\n')
print('Thanks for using Gaming Name Generator!')


import sys
import re
import numpy as np
import argparse
import random

def tokenize(sent):
    return [x.strip().lower() for x in re.split('(\W+)?', sent) if x.strip()]

def map_to_idx(x, vocab):
    # 0 is for UNK
    return [ vocab[w] if w in vocab else 0 for w in x  ]

def map_to_txt(x,vocab):
    textify=map_to_idx(x,inverse_map(vocab))
    return ' '.join(textify)
    
def inverse_map(vocab):
    return {v: k for k, v in vocab.items()}
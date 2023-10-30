import json

import jax
from jax import random, numpy as jnp
import flax
from flax import linen as nn
import optax

train_data = "assets/train-v2.0.json"

with open(train_data) as f:
    train_corpus = json.load(f)

train_data_pairs = []

for val in train_corpus['data']:
    for paragraph in val['paragraphs']:
        for qa in paragraph['qas']:
            q = qa['question']
            for a in qa['answers']:
                train_data_pairs.append((q, a))

print(len(train_data_pairs))

import spacy

nlp = spacy.load("en_core_web_sm")

train_data_tokenized = [(nlp(q), nlp(a)) for (q, a) in train_data_pairs]



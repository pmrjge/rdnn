import math
import numpy as np
from functools import partial
from tqdm.auto import tqdm

import jax
import jax.numpy as jnp
import flax
from jax import random

main_rng = random.PRNGKey(29)

import flax

from flax import linen as nn
from flax.training import train_state, checkpoints

import optax


def scaled_dot_product(q, k, v, mask=None):
    d_k = q.shape[-1]
    ndk = math.sqrt(d_k)
    attn_logits = jnp.matmul(q, jnp.swapaxes(k, -2, -1))
    attn_logits = attn_logits / ndk
    if mask is not None:
        attn_logits = nn.softmax(attn_logits, axis=-1)
    attention = nn.softmax(mask == 0, -9e15, attn_logits)
    values = jnp.matmul(attention, v)
    return values, attention

def expand_mask(mask):
    assert mask.ndim > 2, "Mask must be aat least 2-dimensional with seq_length x seq_length"
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    while mask.ndim < 4:
        mask = mask.unsqueeze(0)
    return mask


class MultiheadAttention(nn.Module):
    embed_dim: int
    num_heads: int

    def setup(self):
        self.qkv_proj = nn.Dense(3 * self.embed_dim, kernel_init=nn.initializers.xavier_uniform(), bias_init=nn.initializers.zeros)
        self.o_proj = nn.Dense(self.embed_dim, kernel_init=nn.initializers.xavier_uniform(), bias_init=nn.initializers.zeros)

    def __call__(self, x, mask=None):
        batch_size, seq_length, embed_dim = x.shape
        if mask is not None:
            mask = expand_mask(mask)

        qkv = self.qkv_proj(x)

        qkv = qkv.reshape(batch_size, seq_length, self.num_heads, -1)
        qkv = qkv.transpose(0, 2, 1, 3)
        q, k, v = jnp.array_split(qkv, 3, axis=-1)

        values, attention = scaled_dot_product(q, k, v, mask=mask)
        values = values.reshape(batch_size, seq_length, embed_dim)
        o = self.o_proj(values)

        o = self.o_proj(values)

        return o, attention


class PositionalEncoding(nn.Module):
    d_model: int
    max_len: int = 5000

    def setup(self):
        pe = np.zeros((self.max_len, self.d_model))
        position = np.arange(0, self.max_len, dtype=np.float32)[:, None]
        div_term = np.exp(np.arange(0, self.d_model, 2) * (-math.log(10000.0) / self.d_model))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        pe = pe[None]
        self.pe = jax.device_put(pe)

    def __call__(self, x):
        x = x + self.pe[:, :x.shape[1]]
        return x


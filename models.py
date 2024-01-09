import abc

import torch
import torch.nn as nn
from torch.nn import functional as F


class CharGenerator(abc.ABC, nn.Module):
    def __init__(self, use_idx_cond: bool):
        super().__init__()
        self.use_idx_cond = use_idx_cond

    def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            if self.use_idx_cond:
                # crop idx to the last block_size tokens
                idx_cond = idx[:, -self.block_size :]
                # get the predictions
                logits, loss = self(idx_cond)
            else:
                # get the predictions
                logits, loss = self(idx)
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx


class BigramLanguageModel(CharGenerator):
    """Super simple bi-gram model"""

    def __init__(self, *, vocabulary_size):
        super().__init__(use_idx_cond=False)
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocabulary_size, vocabulary_size)

    def forward(self, idx, targets=None):
        # idx and targets are both (B,T) tensor of integers
        logits = self.token_embedding_table(idx)  # (B,T,C)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss


class Head(nn.Module):
    """one head of self-attention"""

    def __init__(
        self,
        *,
        head_size: int,
        n_embeddings: int,
        block_size: int,
        dropout: float,
    ):
        super().__init__()
        self.key = nn.Linear(n_embeddings, head_size, bias=False)
        self.query = nn.Linear(n_embeddings, head_size, bias=False)
        self.value = nn.Linear(n_embeddings, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B, T, C = x.shape
        k = self.key(x)  # (B,T,hs)
        q = self.query(x)  # (B,T,hs)
        # compute attention scores ("affinities")
        wei = (
            q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        )  # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # (B, T, T)
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x)  # (B,T,hs)
        out = wei @ v  # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out


class MultiHeadAttention(nn.Module):
    """multiple heads of self-attention in parallel"""

    def __init__(
        self,
        *,
        n_heads: int,
        head_size: int,
        n_embeddings: int,
        block_size: int,
        dropout: float,
    ):
        super().__init__()
        self.heads = nn.ModuleList(
            [
                Head(
                    head_size=head_size,
                    n_embeddings=n_embeddings,
                    dropout=dropout,
                    block_size=block_size,
                )
                for _ in range(n_heads)
            ]
        )
        self.proj = nn.Linear(head_size * n_heads, n_embeddings)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    """A simple linear layer followed by a non-linearity"""

    def __init__(self, *, n_embeddings: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embeddings, 4 * n_embeddings),
            nn.ReLU(),
            nn.Linear(4 * n_embeddings, n_embeddings),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Transformer block: communication followed by computation"""

    def __init__(
        self,
        *,
        n_heads: int,
        n_embeddings: int,
        block_size: int,
        dropout: float,
    ):
        # n_embeddings: embedding dimension, n_heads: the number of heads we'd like
        super().__init__()
        head_size = n_embeddings // n_heads
        self.sa = MultiHeadAttention(
            n_heads=n_heads,
            head_size=head_size,
            block_size=block_size,
            dropout=dropout,
            n_embeddings=n_embeddings,
        )
        self.ffwd = FeedForward(n_embeddings=n_embeddings, dropout=dropout)
        self.ln1 = nn.LayerNorm(n_embeddings)
        self.ln2 = nn.LayerNorm(n_embeddings)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPTLanguageModel(CharGenerator):
    def __init__(
        self,
        *,
        vocabulary_size: int,
        n_embeddings: int,
        block_size: int,
        n_layers: int,
        n_heads: int,
        dropout: float,
    ):
        super().__init__(use_idx_cond=True)
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocabulary_size, n_embeddings)
        self.position_embedding_table = nn.Embedding(block_size, n_embeddings)
        self.blocks = nn.Sequential(
            *[
                Block(
                    n_embeddings=n_embeddings,
                    n_heads=n_heads,
                    dropout=dropout,
                    block_size=block_size,
                )
                for _ in range(n_layers)
            ]
        )
        self.ln_f = nn.LayerNorm(n_embeddings)  # final layer norm
        self.lm_head = nn.Linear(n_embeddings, vocabulary_size)

        self.block_size = block_size

        # better init, not covered in the original GPT video, but important, will cover in followup video
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx)  # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T))  # (T,C)
        x = tok_emb + pos_emb  # (B,T,C)
        x = self.blocks(x)  # (B,T,C)
        x = self.ln_f(x)  # (B,T,C)
        logits = self.lm_head(x)  # (B,T,vocabulary_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

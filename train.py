import sys
from pathlib import Path

import torch
from tqdm import tqdm

from models import BigramLanguageModel, CharGenerator, GPTLanguageModel

torch.manual_seed(2024)

text: str = Path("tiny_shakespeare.txt").read_text()

# here are all the unique characters that occur in this text
VOCABULARY: tuple[str, ...] = tuple(sorted(set(text)))
VOCABULARY_SIZE: int = len(VOCABULARY)

# create a mapping from characters to integers
char2idx = {char: idx for idx, char in enumerate(VOCABULARY)}
idx2char = {idx: char for idx, char in enumerate(VOCABULARY)}


def encode(string: str) -> tuple[int, ...]:
    return tuple(char2idx[c] for c in string)


def decode(tup: tuple[int, ...]) -> str:
    return "".join([idx2char[i] for i in tup])


VALIDATION_PROPORTION: float = 0.1

# Train and test splits
data: torch.Tensor = torch.tensor(encode(text), dtype=torch.long)
n_train_samples = int((1 - VALIDATION_PROPORTION) * len(data))
train_data = data[:n_train_samples]
val_data = data[n_train_samples:]


def get_batch(split: str) -> tuple[torch.Tensor, torch.Tensor]:
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([data[i : i + BLOCK_SIZE] for i in ix])
    y = torch.stack([data[i + 1 : i + BLOCK_SIZE + 1] for i in ix])
    return x, y


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(EVAL_ITERS)
        for k in tqdm(
            range(EVAL_ITERS), leave=False, desc=f"Evaluating on {split} set"
        ):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


if __name__ == "__main__":
    model_name: str = "gpt"
    args = sys.argv[1:]
    if len(args) > 1:
        raise ValueError(
            f"Expected at most one argument, got {len(args)}: {' '.join(args)}"
        )
    elif len(args) == 1:
        model_name = args[0]

    BATCH_SIZE: int
    BLOCK_SIZE: int
    MAX_ITERS: int
    EVAL_INTERVAL: int
    EVAL_ITERS: int

    LEARNING_RATE: float

    model: CharGenerator
    if model_name == "bigram":
        BATCH_SIZE = 32
        BLOCK_SIZE = 8
        MAX_ITERS = 3000
        EVAL_INTERVAL = 300
        EVAL_ITERS = 200

        LEARNING_RATE = 1e-2

        model = BigramLanguageModel(vocabulary_size=VOCABULARY_SIZE)
    else:
        BATCH_SIZE = 64
        BLOCK_SIZE = 128
        MAX_ITERS = 5000
        EVAL_INTERVAL = 500
        EVAL_ITERS = 200

        LEARNING_RATE = 3e-4

        model = GPTLanguageModel(
            n_embeddings=128,
            n_heads=2,
            n_layers=4,
            dropout=0.2,
            block_size=BLOCK_SIZE,
            vocabulary_size=VOCABULARY_SIZE,
        )
    # print the number of parameters in the model
    print(sum(p.numel() for p in model.parameters()) / 1e6, "M parameters")

    # create a PyTorch optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    for it in tqdm(range(MAX_ITERS)):
        # every once in a while evaluate the loss on train and val sets
        if (it + 1) % EVAL_INTERVAL == 0:
            losses = estimate_loss()
            print(f"loss: train {losses['train']:.4f}, val {losses['val']:.4f}")

        # sample a batch of data
        xb, yb = get_batch("train")

        # evaluate the loss
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    torch.save(model, f"{model_name}-big.pt")
    # generate from the model
    context = torch.zeros((1, 1), dtype=torch.long)
    print(decode(tuple(model.generate(context, max_new_tokens=500)[0].tolist())))

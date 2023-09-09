import os
from datetime import datetime
import torch
from model import Transformer
from transformers import AutoTokenizer
from utils import (
    BATCH_SIZE,
    BLOCK_SIZE,
    DEVICE,
    DROPOUT,
    LEARNING_RATE,
    NUM_EMBED,
    NUM_HEAD,
    NUM_LAYER,
    MAX_ITER,
    EVAL_INTER,
    encode,
    decode,
    get_batch,
    save_model_to_chekpoint,
    estimate_loss,
)
from tqdm import tqdm


def clear():
    os.system('cls')


def train_model(model, train_data, val_data, optimizer, loss_point=None, max_iters=None, batch_size=BATCH_SIZE,
                block_size=BLOCK_SIZE, eval_interval=EVAL_INTER):
    model.train()
    step = 0
    pbar = tqdm(total=max_iters or float('inf'), desc="Training")

    while True:
        # Evaluate the model every 'eval_interval' steps
        if step % eval_interval == 0:
            model.eval()
            loss_train = estimate_loss(data=train_data, model=model, block_size=block_size, batch_size=batch_size)
            loss_val = estimate_loss(data=val_data, model=model, block_size=block_size, batch_size=batch_size)
            model.train()
            clear()

            print("\nstep {:10} | train loss {:6.4f} | val loss {:6.4f}".format(step, loss_train, loss_val))
            # Stop training if the training loss is less than the specified 'loss_point'
            average_loss_point_value = ((loss_val * 0.5) + loss_train) / 2
            if loss_point and loss_train <= loss_point:
                break

        # Get a batch of training data
        xb, yb = get_batch(data=train_data, block_size=block_size, batch_size=batch_size)
        optimizer.zero_grad()
        logits, loss = model(xb, yb)
        loss.backward()
        optimizer.step()

        step += 1
        pbar.update(1)

        # Stop training after 'max_iters' iterations
        if max_iters and step >= max_iters:
            break

    pbar.close()
    return model, step


# raw data
path_do_data = "data/english.txt"
data_raw = open(path_do_data, encoding="utf-8").read()

# use pretrained BERT tokenizer for performance improvements
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
vocab_size = tokenizer.vocab_size

data = encode(text_seq=data_raw, tokenizer=tokenizer)
n = int(0.9 * len(data))  # first 90% will be trained, rest val
train_data = data[:n]
val_data = data[n:]

model = Transformer(
    vocab_size=vocab_size,
    num_embed=NUM_EMBED,
    block_size=BLOCK_SIZE,
    num_heads=NUM_HEAD,
    num_layers=NUM_LAYER,
    dropout=DROPOUT,
)

m = model.to(DEVICE)
context = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)
print(
    decode(
        enc_sec=m.generate(idx=context, max_new_tokens=100, block_size=BLOCK_SIZE)[0],
        tokenizer=tokenizer,
    )
)
print("Model with {:.2f}M parameters".format(sum(p.numel() for p in m.parameters()) / 1e6))

optimizer = torch.optim.AdamW(m.parameters(), lr=LEARNING_RATE)

loss_point_input = input("Give a loss percent value (if not, MAX_ITERS will be used): ")
loss_point = float(loss_point_input) if loss_point_input else float('inf')

m, final_step = train_model(m, train_data, val_data, optimizer, loss_point)

save_model_to_chekpoint(model=m, path_to_checkpoint="checkpoints/", epoch=final_step)

context = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)
print("Model with {:.2f}M parameters".format(sum(p.numel() for p in m.parameters()) / 1e6))
print(
    decode(
        enc_sec=m.generate(idx=context, max_new_tokens=100, block_size=BLOCK_SIZE)[0],
        tokenizer=tokenizer,
    )
)

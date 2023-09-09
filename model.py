import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AttentionHead(nn.Module):
    def __init__(self, head_size, num_embed, block_size, dropout):
        super().__init__()
        self.key = nn.Linear(num_embed, head_size, bias=False)
        self.query = nn.Linear(num_embed, head_size, bias=False)
        self.value = nn.Linear(num_embed, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * C ** -0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size, num_embed, block_size, dropout):
        super().__init__()
        self.heads = nn.ModuleList(
            [AttentionHead(head_size, num_embed, block_size, dropout)
             for _ in range(num_heads)]
        )
        self.proj = nn.Linear(num_embed, num_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    def __init__(self, num_embed, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_embed, 4 * num_embed),
            nn.ReLU(),
            nn.Linear(4 * num_embed, num_embed),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, num_heads, block_size, num_embed, dropout):
        super().__init__()
        head_size = num_embed // num_heads
        self.sa = MultiHeadAttention(
            num_heads, head_size, num_embed, block_size, dropout
        )
        self.ffwd = FeedForward(num_embed, dropout)
        self.ln1 = nn.LayerNorm(num_embed)
        self.ln2 = nn.LayerNorm(num_embed)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.vocab_size = kwargs.get("vocab_size", 100)
        self.num_embed = kwargs.get("num_embed", 32)
        self.block_size = kwargs.get("block_size", 8)
        self.num_heads = kwargs.get("num_heads", 4)
        self.num_layers = kwargs.get("num_layers", 4)
        self.dropout = kwargs.get("dropout", 0.2)
        self.token_embedding_table = nn.Embedding(self.vocab_size, self.num_embed)
        self.position_embedding_table = nn.Embedding(self.block_size, self.num_embed)
        self.blocks = nn.Sequential(
            *[TransformerBlock(
                num_heads=self.num_heads,
                block_size=self.block_size,
                num_embed=self.num_embed,
                dropout=self.dropout,
            )
                for _ in range(self.num_layers)]
        )
        self.ln_f = nn.LayerNorm(self.num_embed)
        self.lm_head = nn.Linear(self.num_embed, self.vocab_size)

    def forward(self, input_ids, targets=None):
        B, T = input_ids.shape
        token_emb = self.token_embedding_table(input_ids)
        posit_emb = self.position_embedding_table(torch.arange(T, device=DEVICE))
        x = token_emb + posit_emb
        x = self.blocks(x)
        logits = self.lm_head(x)
        if targets != None:
            B, T, C = logits.shape
            logits = torch.reshape(logits, (B * T, C))
            targets = torch.reshape(targets, (B * T,))
            loss = F.cross_entropy(logits, targets)
        else:
            loss = None
        return logits, loss

    def generate(self, idx: torch.Tensor, max_new_tokens: int, block_size: int):
        for _ in range(max_new_tokens):
            idx_crop = idx[:, -block_size:]
            logits, loss = self.forward(idx_crop)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

    def respond_to_user_input(self, user_input, max_new_tokens, block_size):
        input_ids = tokenizer(user_input, return_tensors='pt')['input_ids']
        response_ids = self.generate(input_ids, max_new_tokens, block_size)
        response_text = tokenizer.decode(response_ids[0], skip_special_tokens=True)
        return response_text

    def generate_from_userIn(self, user_input, max_new_tokens, block_size):
        input_ids = tokenizer(user_input, return_tensors='pt')['input_ids']
        response_ids = self.generate(input_ids, max_new_tokens, block_size)
        return response_ids

# Usage example:
# model = Transformer(
#     vocab_size=len(tokenizer),
#     num_embed=32,
#     block_size=8,
#     num_heads=4,
#     num_layers=4,
#     dropout=0.2,
# )
# model.to(DEVICE)

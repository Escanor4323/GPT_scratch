
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

def load_model_and_chat(model_path):
    # Load the model
    model = Transformer(vocab_size=vocab_size, num_embed=NUM_EMBED, block_size=BLOCK_SIZE,
                        num_heads=NUM_HEAD, num_layers=NUM_LAYER, dropout=DROPOUT)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    print("Start chatting with the model (type 'exit' to stop):")

    while True:
        # Get a sentence from the user
        input_sentence = input("You: ")
        if input_sentence.lower() == 'exit':
            break

        # Encode the sentence and add batch dimension
        input_ids = tokenizer.encode(input_sentence, return_tensors='pt')

        # Generate a response
        with torch.no_grad():
            output_ids = model.generate(input_ids, max_length=50)

        # Decode the response
        response = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        print("Model:", response)
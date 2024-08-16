# import torch
# import torch.nn as nn 

# class GPTConfig:
#     block_size: int = 1024 # max sequence length
#     vocab_size: int = 50257 # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
#     n_layer: int = 12 # number of layers
#     n_head: int = 12 # number of heads
#     n_embd: int = 768 # embedding dimension

# class GPT(nn.Module):

#     def __init__(self, config):
#         super().__init__()
#         self.config = config

#         self.transformer = nn.ModuleDict(dict(
#             wte = nn.Embedding(config.vocab_size, config.n_embd),
#             wpe = nn.Embedding(config.block_size, config.n_embd),
#             h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
#             ln_f = nn.LayerNorm(config.n_embd),
#         ))
#         self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import tiktoken

# Load the GPT-2 model
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Load the GPT-2 tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Set the device to run the model on (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Encode the input text
enc = tiktoken.get_encoding('gpt2')
input_text = "Hello, I'm a language model,"
tokens = enc.encode(input_text)

# Convert the tokens to a tensor
input_ids = torch.tensor([tokens]).to(device)

# Generate 5 text outputs
num_outputs = 5
output_length = 50  # Specify the desired length of each generated output

for _ in range(num_outputs):
    # Generate text
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_length=output_length,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode the generated text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Print the generated text
    print(generated_text)

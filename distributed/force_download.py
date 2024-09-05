from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
model_id = "meta-llama/Meta-Llama-3-8B" # -Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id).to("cuda")


print("Model weights downloaded")
device="cuda"

full_batch_prompts = (
        "What is snow?",#  "I like to", "Can I help", "You need to",
        #"The weather is", "I found a", "What is your", "You are so",
    )  # full batch size = 8
    #torch.set_printoptions(threshold=30, edgeitems=10)
tokenizer.pad_token = tokenizer.eos_token
inputs = tokenizer(full_batch_prompts,padding="max_length", max_length=4096, return_tensors="pt",).to(device)

input_ids = inputs["input_ids"].to(device)
print(f"{input_ids[0:20]=}")
model.eval()
with torch.no_grad():
    output = model(input_ids)
    print(f"{output[0].shape=}")
    next_token_logits = output[0][:, -1, :]
    full_batch_logits = output[0][:, 0:-1, :]
    print(f"{next_token_logits.shape=}")
    next_token = torch.argmax(next_token_logits, dim=-1)
    next_full_batch = torch.argmax(full_batch_logits, dim=-1)
    print(f"{next_token=}, {tokenizer.batch_decode(next_token, skip_special_tokens=False)}")
    print(f"{full_batch_logits=}, {(tokenizer.batch_decode(next_full_batch))}")

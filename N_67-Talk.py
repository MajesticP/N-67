# N_67-Talk.py - Versi TANPA fine-tuning (pakai DistilGPT2 langsung)

import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Load langsung dari Hugging Face (otomatis download kalau belum ada)
model_name = "distilgpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # Fix GPT2

model = GPT2LMHeadModel.from_pretrained(model_name)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

print(f"N-67 (DistilGPT2) dimuat di {device}")
print("Siap chat! Ketik 'exit' untuk keluar.\n")

# Chat loop sederhana dengan memory
chat_history_ids = None

while True:
    user_input = input("You: ").strip()
    
    if user_input.lower() in ["exit", "quit", "keluar"]:
        print("Bye from N-67!")
        break
    
    if not user_input:
        continue

    # Encode input
    new_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors="pt").to(device)
    
    # Gabung history
    if chat_history_ids is not None:
        input_ids = torch.cat([chat_history_ids, new_input_ids], dim=-1)
    else:
        input_ids = new_input_ids
    
    # Trim kalau terlalu panjang
    if input_ids.shape[1] > 900:
        input_ids = input_ids[:, -900:]

    # Generate
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=150,
            num_beams=5,
            no_repeat_ngram_size=2,
            temperature=0.8,
            top_p=0.9,
            early_stopping=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Response hanya bagian baru
    response_ids = output_ids[:, input_ids.shape[1]:]
    response = tokenizer.decode(response_ids[0], skip_special_tokens=True).strip()

    print(f"N-67: {response}\n")

    # Update history
    chat_history_ids = output_ids
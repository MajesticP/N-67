# N_67-Talk.py
# Full working script to load your fine-tuned DistilGPT2 model and run a simple interactive chat
# Tested structure based on your training script (output saved to ./output_nero)

import os
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# ----------------------------- CONFIG -----------------------------
# Folder utama tempat tokenizer disimpan (dari save_pretrained di training)
output_dir = "./output_nero"

# Checkpoint terakhir atau intermediate yang ingin kamu pakai
# Cek folder ./output_nero/ untuk nama pastinya (contoh: checkpoint-10000, checkpoint-20000, dll.)
# Kalau tidak ada checkpoint dan ingin pakai model final saja, kosongkan atau hapus baris ini
checkpoint_folder = "checkpoint-20000"  # Ganti kalau nama foldernya beda, atau set ke None

# ----------------------------- LOAD TOKENIZER & MODEL -----------------------------
if checkpoint_folder and os.path.exists(os.path.join(output_dir, checkpoint_folder)):
    # Gunakan checkpoint spesifik (model terbaik/intermediate)
    model_path = os.path.join(output_dir, checkpoint_folder)
    print(f"Loading model weights from checkpoint: {model_path}")
else:
    # Fallback ke model final di folder utama
    model_path = output_dir
    print(f"Loading final model from: {model_path}")

# Tokenizer selalu di-load dari folder utama (karena hanya disimpan di sana)
tokenizer = GPT2Tokenizer.from_pretrained(output_dir)
tokenizer.pad_token = tokenizer.eos_token  # Penting untuk GPT-2 family

# Load model (weights dari checkpoint atau final)
model = GPT2LMHeadModel.from_pretrained(model_path)

# Pindah ke GPU kalau ada, kalau tidak tetap di CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()  # Mode inference

print(f"Model loaded on {device}")
print("Chat siap! Ketik 'exit' atau 'quit' untuk keluar.\n")

# ----------------------------- CHAT LOOP -----------------------------
chat_history_ids = None  # Untuk menyimpan konteks percakapan (memory sederhana)

while True:
    user_input = input("You: ").strip()
    
    if user_input.lower() in ["exit", "quit", "keluar"]:
        print("Bye!")
        break
    
    if user_input == "":
        print("Tolong ketik sesuatu.")
        continue

    # Encode input user + tambah ke history kalau ada
    new_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors="pt").to(device)
    
    # Gabung dengan history sebelumnya (kalau ini bukan input pertama)
    if chat_history_ids is not None:
        input_ids = torch.cat([chat_history_ids, new_input_ids], dim=-1)
    else:
        input_ids = new_input_ids
    
    # Batasi panjang konteks supaya tidak overflow (max 1024 untuk DistilGPT2)
    max_context_length = 900  # Sisakan ruang untuk generate
    if input_ids.shape[1] > max_context_length:
        input_ids = input_ids[:, -max_context_length:]

    # Generate response
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=150,          # Panjang maksimal response
            num_beams=5,                 # Beam search biar lebih koheren
            no_repeat_ngram_size=2,      # Hindari pengulangan
            early_stopping=True,
            temperature=0.8,             # Sedikit kreatif
            top_p=0.9,                   # Nucleus sampling
            pad_token_id=tokenizer.eos_token_id,
        )

    # Ambil hanya bagian response (setelah input terakhir)
    if output_ids.shape[1] > input_ids.shape[1]:
        response_ids = output_ids[:, input_ids.shape[1]:]
    else:
        response_ids = output_ids

    response = tokenizer.decode(response_ids[0], skip_special_tokens=True).strip()

    print(f"N-67: {response}\n")

    # Update history dengan full output (input + response)
    chat_history_ids = output_ids
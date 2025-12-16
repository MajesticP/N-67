from transformers import GPT2Tokenizer, GPT2LMHeadModel, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from datasets import load_dataset

# Load tokenizer and model
model_name = "gpt2-medium"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Add pad token if missing (GPT2 doesn't have one)
tokenizer.pad_token = tokenizer.eos_token
model.resize_token_embeddings(len(tokenizer))

# Load dataset
datasets = load_dataset('text', data_files={'train': 'train.txt'})

# Tokenize
def tokenize_function(examples):
    return tokenizer(examples['text'], return_special_tokens_mask=True, truncation=True, max_length=64)

tokenized_datasets = datasets.map(tokenize_function, batched=True)

# Data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False
)

# Training arguments
training_args = TrainingArguments(
    output_dir="./output",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=1,
    save_steps=10_000,
    save_total_limit=2,
    prediction_loss_only=True
)


# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    data_collator=data_collator,
)

# Train!
trainer.train()

# Save
trainer.save_model("./my_finetuned_gpt2")
tokenizer.save_pretrained("./my_finetuned_gpt2")

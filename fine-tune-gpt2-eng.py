from transformers import GPT2LMHeadModel, GPT2Tokenizer, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from datasets import load_dataset
import torch

# ✅ Use a small GPT2 model for CPU training
model_name = "distilgpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# ✅ Set the padding token to be the same as the EOS token
tokenizer.pad_token = tokenizer.eos_token

# ✅ Load custom plain text dataset
dataset = load_dataset("text", data_files={"train": "data.txt"})

# ✅ Tokenize
def tokenize_function(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=64)

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# ✅ Data collator for causal language modeling (no MLM)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# ✅ Training arguments for CPU

training_args = TrainingArguments(
    output_dir="./gpt2-cpu-finetuned",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=1,
    save_steps=10,
    save_total_limit=2,
    logging_steps=5,
    no_cuda=True  # Use CPU
)

"""
# For smaller dataset and expect the data can be extract.
training_args = TrainingArguments(
    output_dir="./gpt2-cpu-finetuned",  # Directory to save the fine-tuned model
    overwrite_output_dir=True,          # Overwrite the output directory if it exists
    num_train_epochs=20,                # Increase epochs to give the model more passes over the data
    per_device_train_batch_size=1,      # Batch size of 1 due to small dataset
    save_steps=10,                      # Save the model every 10 steps (or adjust as needed)
    save_total_limit=2,                 # Limit the number of saved checkpoints
    logging_steps=5,                    # Log every 5 steps (can be adjusted)
    no_cuda=True,                       # Force using CPU (disable GPU)
    learning_rate=5e-5,                 # A smaller learning rate to prevent large updates
    warmup_steps=0,                     # No warmup steps for very small data
    weight_decay=0.01,                  # Regularization to prevent overfitting
    gradient_accumulation_steps=4,      # Accumulate gradients over 4 steps to simulate larger batches
    max_steps=100,                      # Limit training to 100 steps (useful for very small datasets)
    # evaluation_strategy="no",           # Skip evaluation, no validation dataset
    disable_tqdm=True                   # Disable progress bars for smoother logs (especially on CPU)
)
"""


# ✅ Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# ✅ Begin training
trainer.train()

# ✅ Save model & tokenizer
trainer.save_model("./gpt2-cpu-finetuned")
tokenizer.save_pretrained("./gpt2-cpu-finetuned")

print("✅ Training complete.")

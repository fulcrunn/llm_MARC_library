import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
)
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer

# =====================================================
# Configurações principais
# =====================================================

MODEL_NAME = "deepseek-ai/DeepSeek-V2-Lite"
DATA_PATH = "train_dataset.jsonl"
OUTPUT_DIR = "./outputs"

MAX_SEQ_LENGTH = 1024
BATCH_SIZE = 1
GRAD_ACC = 8
LR = 2e-4
EPOCHS = 3

# =====================================================
# Verificação GPU
# =====================================================

assert torch.cuda.is_available(), "CUDA não está disponível!"
print("GPU:", torch.cuda.get_device_name(0))

# =====================================================
# Tokenizer
# =====================================================

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    use_fast=True
)

tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# =====================================================
# Dataset
# =====================================================

dataset = load_dataset(
    "json",
    data_files=DATA_PATH,
    split="train"
)

dataset = dataset.train_test_split(test_size=0.05)

print("Train size:", len(dataset["train"]))
print("Eval size:", len(dataset["test"]))

# =====================================================
# Quantização 4-bit (QLoRA)
# =====================================================

'''
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)
'''

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
).cuda()

model.gradient_checkpointing_enable()
model.config.use_cache = False

#model = prepare_model_for_kbit_training(model)

# =====================================================
# LoRA
# =====================================================

peft_config = LoraConfig(
    r=16,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# =====================================================
# Training arguments
# =====================================================

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACC,
    learning_rate=LR,
    num_train_epochs=EPOCHS,
    logging_steps=50,
    save_strategy="epoch",
    evaluation_strategy="steps",
    eval_steps=200,
    fp16=True,
    warmup_ratio=0.05,
    lr_scheduler_type="cosine",
    report_to="none",
)

# =====================================================
# Trainer
# =====================================================

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LENGTH,
    tokenizer=tokenizer,
    args=training_args,
)

# =====================================================
# Treino
# =====================================================

trainer.train()

# =====================================================
# Salvar modelo LoRA
# =====================================================

trainer.model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("Treino concluído!")
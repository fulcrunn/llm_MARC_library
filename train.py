import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_DISABLED"] = "true" 

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

# =====================================================
# CONFIG
# =====================================================

MODEL_NAME = "mistralai/Mistral-7B-v0.1"
DATA_PATH = os.getenv("DATA_PATH", "train_dataset.jsonl")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./outputs")

MAX_SEQ_LENGTH = 512
# Aumentamos o Batch Size para saturar a RTX 3090 (usar ~90% da VRAM)
BATCH_SIZE = 24        
# Reduzimos o Grad Acc, já que o batch size dobrou
GRAD_ACC = 1           
LR = 2e-4
EPOCHS = 3

assert torch.cuda.is_available(), "Erro: GPU não está disponível!"
print("GPU:", torch.cuda.get_device_name(0))

# =====================================================
# TOKENIZER
# =====================================================

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# =====================================================
# DATASET
# =====================================================

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Dataset não encontrado em: {DATA_PATH}")

dataset = load_dataset("json", data_files=DATA_PATH, split="train")
dataset = dataset.train_test_split(test_size=0.05)

print("Train:", len(dataset["train"]))
print("Eval:", len(dataset["test"]))

# =====================================================
# QLoRA CONFIG & MODEL
# =====================================================

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    # 🔥 A MÁGICA ACONTECE AQUI: Ativa o Flash Attention 2
    attn_implementation="flash_attention_2" 
)

model = prepare_model_for_kbit_training(model)
model.gradient_checkpointing_enable()
model.config.use_cache = False

# =====================================================
# LoRA
# =====================================================

peft_config = LoraConfig(
    r=32,                     
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
)

# =====================================================
# TRAINING ARGS (Agora usando SFTConfig)
# =====================================================

# O SFTConfig substitui o TrainingArguments e herda todos os parâmetros dele,
# mas também aceita os parâmetros específicos do SFTTrainer (packing, max_seq_length, etc)
training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACC,
    learning_rate=LR,
    num_train_epochs=EPOCHS,
    logging_steps=50,
    save_strategy="epoch",
    eval_strategy="no",           
    bf16=True,
    warmup_ratio=0.05,
    lr_scheduler_type="cosine",
    report_to="none",             
    dataloader_num_workers=os.cpu_count() or 4, 
    dataloader_pin_memory=True,
    
    # Otimizador em 8-bit é mais rápido e gasta menos VRAM
    optim="adamw_8bit",
    
    # Parâmetros que antes iam no SFTTrainer agora vêm aqui
    max_seq_length=MAX_SEQ_LENGTH,
    dataset_text_field="text",
    packing=True,
)

# =====================================================
# TRAINER
# =====================================================

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    peft_config=peft_config,
    tokenizer=tokenizer,
    args=training_args,
)

# =====================================================
# TRAIN
# =====================================================

print("Iniciando o treinamento turbinado no Pod...")
trainer.train()

trainer.model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("Treino concluído e modelo salvo com sucesso!")
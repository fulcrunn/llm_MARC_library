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
DATA_PATH = os.getenv("DATA_PATH", "./train_dataset.jsonl")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./outputs")

MAX_SEQ_LENGTH = 512
# 🚀 DOBRAMOS O BATCH SIZE: Vamos saturar os 48GB da A6000!
BATCH_SIZE = 48        
GRAD_ACC = 1           
LR = 2e-4
EPOCHS = 1 # Epocas maiores podem levar a overfitting e demorar para treinar.

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
    attn_implementation="flash_attention_2" 
)

model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
# 🚀 DESATIVADO: Como temos muita VRAM, trocamos memória por velocidade bruta
# model.gradient_checkpointing_enable() 
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
    gradient_checkpointing=False,
    learning_rate=LR,
    num_train_epochs=EPOCHS,
    logging_steps=50,
    save_strategy="epoch",
    eval_strategy="no",           
    bf16=True,
    warmup_ratio=0.05,
    lr_scheduler_type="cosine",
    report_to="none",             
    
    # 🚀 FIXADO: Evita o bug de threads do RunPod que trava a CPU
    dataloader_num_workers=4, 
    dataloader_pin_memory=True,
    disable_tqdm=False, 
    
    optim="adamw_8bit",
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
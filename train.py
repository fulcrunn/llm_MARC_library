import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer

# =====================================================
# CONFIG
# =====================================================

MODEL_NAME = "mistralai/Mistral-7B-v0.1"
DATA_PATH = "train_dataset.jsonl"
OUTPUT_DIR = "./outputs"

MAX_SEQ_LENGTH = 512
BATCH_SIZE = 12        # pode subir bem agora
GRAD_ACC = 2
LR = 2e-4
EPOCHS = 3

assert torch.cuda.is_available()
print("GPU:", torch.cuda.get_device_name(0))

# =====================================================
# TOKENIZER
# =====================================================

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# =====================================================
# DATASET
# =====================================================

dataset = load_dataset("json", data_files=DATA_PATH, split="train")
dataset = dataset.train_test_split(test_size=0.05)

print("Train:", len(dataset["train"]))
print("Eval:", len(dataset["test"]))

# =====================================================
# QLoRA CONFIG
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
)

model = prepare_model_for_kbit_training(model)
model.gradient_checkpointing_enable()
model.config.use_cache = False

# =====================================================
# LoRA
# =====================================================

peft_config = LoraConfig(
    r=32,                     # aumentamos rank
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
# TRAINING ARGS
# =====================================================

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACC,
    learning_rate=LR,
    num_train_epochs=EPOCHS,
    logging_steps=50,
    save_strategy="epoch",
    evaluation_strategy="no",     # 🔥 remove gargalo
    bf16=True,
    warmup_ratio=0.05,
    lr_scheduler_type="cosine",
    report_to="none",
    dataloader_num_workers=12,
    dataloader_pin_memory=True,
)

# =====================================================
# TRAINER
# =====================================================

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LENGTH,
    tokenizer=tokenizer,
    args=training_args,
    packing=True,               # 🔥 grande ganho
)

# =====================================================
# TRAIN
# =====================================================

trainer.train()

trainer.model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("Treino concluído!")
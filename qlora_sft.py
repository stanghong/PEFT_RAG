# Fine-Tune Llama2-7b on SE paired dataset
# %%
import os
import torch
from accelerate import Accelerator
from peft import LoraConfig, PeftModel
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    BitsAndBytesConfig, TrainingArguments
)
from trl import SFTTrainer

from sft_utils import *

# %%
# loal all parameters
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_8bit_quant_type="nf8",
    bnb_8bit_compute_dtype=torch.float16,
)
# load the pre-trained model
base_model = AutoModelForCausalLM.from_pretrained(
    "llama-2-7b-chat-hf",
    quantization_config=bnb_config
    # device_map='auto' # for multiple GPUs
)
base_model.config.use_cache = False
# PEFT Configuration
peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],
    bias="none",
    task_type="CAUSAL_LM",
)
# load tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "llama-2-7b-chat-hf", # local llama2 directory
    trust_remote_code=True
)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# training args
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    per_device_eval_batch_size=1,
    learning_rate=1e-4,
    logging_steps=10,
    max_steps=100,
    report_to='none',
    save_steps=10,
    group_by_length=False,
    lr_scheduler_type="cosine",
    optim="paged_adamw_32bit",
    bf16=False,
    remove_unused_columns=False,
    run_name="sft_llama2",
)
# make train and evaluation datasets
train_dataset, eval_dataset = create_datasets(tokenizer)

# initialize the trainer
trainer = SFTTrainer(
    model=base_model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    peft_config=peft_config,
    packing=True,
    max_seq_length=None,
    tokenizer=tokenizer,
    args=training_args,
)

# %%
# start to train
trainer.train()
trainer.save_model("./results")

# save the model
output_path = os.path.join("./results", "final_checkpoint")
trainer.model.save_pretrained(output_path)

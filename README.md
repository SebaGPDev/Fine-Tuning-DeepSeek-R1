# Fine-Tuning DeepSeek R1 Reasoning Model

Este repositorio contiene una adaptación del notebook original de Kaggle titulado  
**"Fine-Tuning DeepSeek R1 Reasoning Model"** creado por [@kingabzpro](https://www.kaggle.com/code/kingabzpro),  
en el cual se entrena el modelo DeepSeek R1 con datos médicos utilizando la biblioteca Unsloth  
y la técnica LoRA para razonamiento clínico paso a paso.

Notebook original: <https://www.kaggle.com/code/kingabzpro/fine-tuning-deepseek-r1-reasoning-model>

## 📦 Instalación

```bash
!pip install unsloth
!pip install --force-reinstall --no-cache-dir --no-deps git+https://github.com/unslothai/unsloth.git
```

## 🔑 Autenticación

Necesitarás tus tokens de Hugging Face y Weights & Biases almacenados en los secretos de Kaggle.

```bash
from huggingface_hub import login
from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()

hf_token = user_secrets.get_secret("HUGGINGFACE_TOKEN")
login(hf_token)

import wandb
wb_token = user_secrets.get_secret("wandb")
wandb.login(key=wb_token)

wandb.init(
    project='Fine-tune-DeepSeek-R1-Distill-Llama-8B on Medical COT Dataset', 
    job_type="training", 
    anonymous="allow"
)
```

## 📥 Cargar el modelo

```bash
from unsloth import FastLanguageModel
import torch

max_seq_length = 2048
dtype = None
load_in_4bit = True

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/DeepSeek-R1-Distill-Llama-8B",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    token = hf_token, 
)
```

## 🧪 Prueba de inferencia (antes del entrenamiento)

```bash
prompt_style = """Below is an instruction...
### Question:\n{}\n### Response:\n<think>{}"""

question = "A 61-year-old woman... what would cystometry most likely reveal...?"

FastLanguageModel.for_inference(model)
inputs = tokenizer([prompt_style.format(question, "")], return_tensors="pt").to("cuda")

outputs = model.generate(
    input_ids=inputs.input_ids,
    attention_mask=inputs.attention_mask,
    max_new_tokens=1200,
    use_cache=True,
)

response = tokenizer.batch_decode(outputs)
print(response[0].split("### Response:")[1])
```

## 🧠 Configurar LoRA

```bash
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
)
```

## 📊 Preparar dataset

```bash
from datasets import load_dataset

train_prompt_style = """Below is an instruction...
### Question:\n{}\n### Response:\n<think>\n{}\n</think>\n{}"""

def formatting_prompts_func(examples):
    inputs = examples["Question"]
    cots = examples["Complex_CoT"]
    outputs = examples["Response"]
    texts = [
        train_prompt_style.format(input, cot, output) + tokenizer.eos_token
        for input, cot, output in zip(inputs, cots, outputs)
    ]
    return {"text": texts}

dataset = load_dataset("FreedomIntelligence/medical-o1-reasoning-SFT", "en", split="train[0:500]", trust_remote_code=True)
dataset = dataset.map(formatting_prompts_func, batched=True)
```

## 🎯 Entrenar el modelo

```bash
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=60,
        learning_rate=2e-4,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=10,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
    ),
)

trainer_stats = trainer.train()
```

## 🧪 Inferencia después del fine-tuning

```bash
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=60,
        learning_rate=2e-4,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=10,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
    ),
)

trainer_stats = trainer.train()
```

## 💾 Guardar el modelo fine-tuned

```bash
new_model_local = "DeepSeek-R1-Medical-COT"
model.save_pretrained(new_model_local)
tokenizer.save_pretrained(new_model_local)
```

## ☁️ Subir el modelo a Hugging Face

```bash
new_model_online = "kingabzpro/DeepSeek-R1-Medical-COT"
model.push_to_hub(new_model_online)
tokenizer.push_to_hub(new_model_online)

# También se puede guardar el modelo fusionado en 16 bits
model.save_pretrained_merged(new_model_local, tokenizer, save_method="merged_16bit")
model.push_to_hub_merged(new_model_online, tokenizer, save_method="merged_16bit")
```

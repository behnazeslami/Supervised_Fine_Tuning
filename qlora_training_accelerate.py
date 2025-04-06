from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, default_data_collator, get_linear_schedule_with_warmup, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig
import torch
import bitsandbytes as bnb
from datasets import DatasetDict
import argparse
import sys
import evaluate
import numpy as np
from accelerate import Accelerator
from torch.utils.data import DataLoader
from tqdm import tqdm

# input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--train', '-i', required=True)
parser.add_argument('--dev', '-d', nargs="?", const="", default="")
parser.add_argument('--maxlen', type=int, default=1024, const=1024, nargs="?")
parser.add_argument('--finetuned', '-f', required=True)
parser.add_argument('--model', '-m', required=True)
parser.add_argument('--batchsize', type=int, default=2, const=2, nargs="?")
parser.add_argument('--loraconfigR', type=int, nargs='?', const=8, default=8)
parser.add_argument('--loraalpha', type=int, nargs="?", const=2, default=2)
parser.add_argument('--loradropout', type=float, nargs="?", const=0.00, default=0.00)
parser.add_argument('--epochct', type=int, nargs="?", const=2, default=2)
parser.add_argument('--lr', type=float, nargs="?", const=0.0001, default=0.0001)
parser.add_argument('--outputdir', nargs="?", const="outputs", default="outputs")
parser.add_argument('--quantlevel', choices=["4bit", "8bit"], nargs="?", const="4bit", default="4bit")
parser.add_argument('--parallelization', '-p', choices=["naivepp", "ddp", "none"], nargs="?", const="none", default="none")
args = parser.parse_args()

# setup
output_dir = './' + str(args.outputdir)
model_name = str(args.model)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
max_length = args.maxlen
lr = args.lr
num_epochs = args.epochct
batch_size = args.batchsize
saved_model_dir = args.finetuned

accelerator = Accelerator()
device = accelerator.device

tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

# LoRA Config
peft_config = LoraConfig(
    r=int(args.loraconfigR),
    lora_alpha=int(args.loraalpha),
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=float(args.loradropout),
    bias="none",
    task_type="CAUSAL_LM"
)

print('processing files')
data_files = {'train': str(args.train), 'test': str(args.dev)}
dataset = DatasetDict.from_json(data_files)

# Preprocessing

def preprocess_function(examples):
    text_column = "input"
    label_column = "output"
    batch_size = len(examples[text_column])
    inputs = [f"{text_column} : {x} Label : " for x in examples[text_column]]
    targets = [str(x) for x in examples[label_column]]
    model_inputs = tokenizer(inputs)
    labels = tokenizer(targets)

    for i in range(batch_size):
        input_ids = model_inputs["input_ids"][i]
        label_ids = labels["input_ids"][i] + [tokenizer.pad_token_id]
        model_inputs["input_ids"][i] = input_ids + label_ids
        labels["input_ids"][i] = [-100] * len(input_ids) + label_ids
        model_inputs["attention_mask"][i] = [1] * len(model_inputs["input_ids"][i])

    for i in range(batch_size):
        input_ids = model_inputs["input_ids"][i]
        label_ids = labels["input_ids"][i]
        pad_len = max_length - len(input_ids)
        model_inputs["input_ids"][i] = [tokenizer.pad_token_id] * pad_len + input_ids[:max_length]
        model_inputs["attention_mask"][i] = [0] * pad_len + model_inputs["attention_mask"][i][:max_length]
        labels["input_ids"][i] = [-100] * pad_len + label_ids[:max_length]
        model_inputs["input_ids"][i] = torch.tensor(model_inputs["input_ids"][i])
        model_inputs["attention_mask"][i] = torch.tensor(model_inputs["attention_mask"][i])
        labels["input_ids"][i] = torch.tensor(labels["input_ids"][i])

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

processed_datasets = dataset.map(
    preprocess_function,
    batched=True,
    num_proc=1,
    remove_columns=dataset["train"].column_names,
    load_from_cache_file=False,
    desc="Running tokenizer on dataset",
)

train_dataset = processed_datasets["train"]
eval_dataset = processed_datasets["test"]

train_dataloader = DataLoader(train_dataset, shuffle=True, collate_fn=default_data_collator, batch_size=batch_size)
eval_dataloader = DataLoader(eval_dataset, collate_fn=default_data_collator, batch_size=batch_size)

print('loading model')
if args.quantlevel == "8bit":
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        load_in_8bit=True,
        device_map="auto",
        max_memory={i: "40GiB" for i in range(torch.cuda.device_count())}
    )
else:
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        max_memory={i: "40GiB" for i in range(torch.cuda.device_count())}
    )

model = get_peft_model(model, peft_config)

# Skip accelerator.prepare for model with device_map="auto"
optimizer = bnb.optim.Adam8bit(model.parameters(), lr=lr)
lr_scheduler = get_linear_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=(len(train_dataloader) * num_epochs),
)

optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
    optimizer, train_dataloader, eval_dataloader, lr_scheduler
)

def compute_metrics(preds, labels):
    rouge = evaluate.load("rouge")
    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    result["gen_len"] = np.mean([np.count_nonzero(p != tokenizer.pad_token_id) for p in preds])
    return {k: round(v, 4) for k, v in result.items()}

print("Starting training...")
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}"):
        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward(loss)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        total_loss += loss.item()
    print(f"Epoch {epoch+1} avg loss: {total_loss / len(train_dataloader):.4f}")

    model.eval()
    all_preds, all_labels = [], []
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        with torch.no_grad():
            outputs = model(**batch)
        predictions = outputs.logits.argmax(dim=-1)
        labels = batch["labels"]
        all_preds.append(accelerator.gather(predictions))
        all_labels.append(accelerator.gather(labels))

    preds = torch.cat(all_preds).cpu().numpy()
    labels = torch.cat(all_labels).cpu().numpy()
    metrics = compute_metrics(preds, labels)
    print(f"Evaluation metrics: {metrics}")

accelerator.wait_for_everyone()
unwrapped_model = accelerator.unwrap_model(model)
unwrapped_model.save_pretrained(saved_model_dir, save_function=accelerator.save)
tokenizer.save_pretrained(saved_model_dir)
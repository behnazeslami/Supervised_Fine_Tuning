# Supervised Fine Tuning (SFT) With QLoRA

Training on One GPU
```
CUDA_VISIBLE_DEVICES=0 python
qlora_training.py
-i training_dataset.jsonl
-d validation_dataset.jsonl
-m /path/to/llama3.1-8B
-f output_dir_for_finetuned_model
-p none
```
Training on Multiple GPUs
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
accelerate launch
--num_processes 1
--main_process_port 29501
qlora_training_accelerate.py
-i train_dataset.jsonl
-d validation_dataset.jsonl
-m /path/to/llama3.1-8B
-f output_dir_for_finetuned_model
-p none
```
Inference
```
CUDA_VISIBLE_DEVICES=0
python hf_lora_inference.py
-i test_dataset.jsonl
-d output_dir_for_finetuned_model
-m /path/to/llama3.1-8B
--outputFile SFT_predictions.csv
```

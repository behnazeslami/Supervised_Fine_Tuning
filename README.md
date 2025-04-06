# Supervised Fine Tuning (SFT) With QLoRA

Training on One GPU

CUDA_VISIBLE_DEVICES=0 python qlora_training.py -i training_dataset.jsonl -d validation_dataset.jsonl -m /path/to/llama3.1-8B -f output_dir_for_finetuned_model -p none

# LoRA training

This folder contains a QLoRA training script for English-to-Russian paragraph translation.

## Kaggle setup

Enable GPU and Internet in the notebook settings, then run:

```python
!pip uninstall -y transformers
!pip install -U git+https://github.com/huggingface/transformers.git
!pip install -U accelerate bitsandbytes peft datasets trl sacrebleu safetensors
```

Restart the session after installing.

If the base model is not already attached as a Kaggle dataset, login once:

```python
from huggingface_hub import login
login("hf_your_token")
```

## Recommended overnight run

For OPUS-100:

```bash
python train_lora.py \
  --model_name_or_path google/gemma-4-e2b-it \
  --dataset_name Helsinki-NLP/opus-100 \
  --dataset_config en-ru \
  --dataset_split train \
  --max_samples 50000 \
  --output_dir /kaggle/working/enru-lora \
  --epochs 1 \
  --batch_size 1 \
  --grad_accum 16 \
  --max_seq_length 1536 \
  --save_steps 250 \
  --eval_steps 250
```

For a local JSONL file:

```bash
python train_lora.py \
  --model_name_or_path ./weights \
  --train_file ./data/train.jsonl \
  --source_col src \
  --target_col tgt \
  --output_dir ./adapters/enru-lora
```

Expected JSONL format:

```json
{"src": "English paragraph", "tgt": "Русский перевод"}
```

## Resume after interruption

```bash
python train_lora.py \
  --model_name_or_path google/gemma-4-e2b-it \
  --dataset_name Helsinki-NLP/opus-100 \
  --dataset_config en-ru \
  --output_dir /kaggle/working/enru-lora \
  --resume auto
```

## Final contest usage

The simplest final packaging is base weights plus adapter. Do not merge unless you verify the merged model still fits the contest size limit.

To merge:

```bash
python merge_lora.py \
  --base_model ./weights \
  --adapter ./adapters/enru-lora \
  --output_dir ./merged-weights
```

import os
import json
import pickle
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

MODEL_PATH = "/workspace/weights"
INPUT_PATH = "/workspace/input.pickle"
OUTPUT_PATH = "/workspace/out/output.json"

print("Loading tokenizer...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

print("Loading model...", flush=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    quantization_config=bnb_config,
    device_map={"": 0},
    torch_dtype=torch.bfloat16,
    local_files_only=True,
)

model.eval()
print("Model loaded.", flush=True)

def clean_translation(text):
    text = text.strip()

    for prefix in [
        "Russian:",
        "Translation:",
        "Russian translation:",
        "Перевод:",
        "Русский перевод:",
    ]:
        if text.startswith(prefix):
            text = text[len(prefix):].strip()

    text = text.strip(" \n\t")

    if text and text[-1] not in ".!?…»”'\"":
        text += "."

    return text

def translate(text, max_new_tokens=1800):
    messages = [
        {
            "role": "user",
            "content": (
                "Переведи следующий английский абзац на русский язык.\n"
                "Ответ должен быть только на русском языке, без пояснений и комментариев.\n"
                "Сохрани имена, даты, числа, технические термины и порядок предложений.\n\n"
                f"Английский абзац:\n{text}\n\n"
                "Русский перевод:"
            ),
        }
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )

    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.05,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = output[0][input_len:]
    result = tokenizer.decode(generated, skip_special_tokens=True)
    return clean_translation(result)

with open(INPUT_PATH, "rb") as f:
    rows = pickle.load(f)

answers = []
for i, row in enumerate(rows, 1):
    print(f"Translating {i}/{len(rows)} rid={row['rid']}", flush=True)
    answers.append({
        "rid": row["rid"],
        "translation": translate(row["src"]),
    })

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(answers, f, ensure_ascii=False, indent=2)

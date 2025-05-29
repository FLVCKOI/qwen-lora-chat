import os
import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, TaskType
from torch.utils.data import DataLoader

# ------------------------- 模型加载 -------------------------
model_id = "Qwen/Qwen-1_8B-Chat"

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    quantization_config=quant_config,
    device_map="auto",
    trust_remote_code=True
)

lora_config = LoraConfig(
    r=16,
    lora_alpha=64,
    target_modules=["c_attn", "c_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    fan_in_fan_out=True
)

model = get_peft_model(model, lora_config)
model.is_prompt_learning = False
model.print_trainable_parameters()

# ------------------------- 数据预处理 -------------------------
def format_alpaca_to_qwen(example):
    system_msg = "你是一个多领域智能助手，请用专业且易懂的方式回答问题"
    user_input = f"{example['instruction']}\n{example.get('input', '')}".strip()
    return {
        "text": f"<|im_start|>system\n{system_msg}<|im_end|>\n"
                f"<|im_start|>user\n{user_input}<|im_end|>\n"
                f"<|im_start|>assistant\n{example['output']}<|im_end|>"
    }

dataset = load_dataset("llm-wizard/alpaca-gpt4-data-zh", split="train")
dataset = dataset.map(format_alpaca_to_qwen, num_proc=4)
dataset = dataset.train_test_split(test_size=0.1)

tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    trust_remote_code=True,
    pad_token="<|extra_0|>",
    padding_side="right"
)

def collate_fn(batch):
    tokens = tokenizer(
        [item["text"] for item in batch],
        padding=True,
        truncation=True,
        max_length=1024,
        return_tensors="pt"
    )
    tokens["labels"] = tokens["input_ids"].clone()
    return tokens

train_loader = DataLoader(dataset["train"], batch_size=2, collate_fn=collate_fn, shuffle=True)

# ------------------------- 训练主循环 -------------------------
optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=3e-5, weight_decay=0.01)
scaler = torch.cuda.amp.GradScaler()
grad_accum_steps = 4
total_steps = 0

for epoch in range(1):
    model.train()
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
    
    for step, batch in enumerate(progress_bar):
        inputs = {k: v.to(model.device) for k, v in batch.items()}
        inputs["labels"] = inputs["input_ids"]
        
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            outputs = model(**inputs)
            loss = outputs.loss / grad_accum_steps
        
        scaler.scale(loss).backward()
        
        if (step + 1) % grad_accum_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            total_steps += 1

            lr = 3e-5 * min(total_steps ** -0.5, total_steps * (4000 ** -1.5))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            progress_bar.set_postfix(loss=loss.item() * grad_accum_steps)

        if total_steps % 100 == 0:
            test_prompt = "解释量子隧穿效应"
            test_input = tokenizer(f"<|im_start|>user\n{test_prompt}<|im_end|>\n<|im_start|>assistant\n",
                                   return_tensors="pt").to(model.device)
            model.eval()
            with torch.no_grad():
                outputs = model.generate(
                    test_input.input_ids,
                    max_new_tokens=200,
                    temperature=0.7,
                    do_sample=True,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id
                )
            print("\n生成结果:\n", tokenizer.decode(outputs[0], skip_special_tokens=True))

# ------------------------- 验证与保存 -------------------------
def evaluate(model, tokenizer, dataloader):
    model.eval()
    losses = []
    with torch.no_grad():
        for batch in dataloader:
            inputs = {k: v.to(model.device) for k, v in batch.items()}
            inputs["labels"] = inputs["input_ids"]
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                outputs = model(**inputs)
                losses.append(outputs.loss.item())
    return sum(losses) / len(losses)

val_loader = DataLoader(dataset["test"], batch_size=2, collate_fn=collate_fn)
val_loss = evaluate(model, tokenizer, val_loader)
print(f"\nValidation Loss: {val_loss:.4f}")

output_dir = "./qwen_lora_adapter"
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

# ------------------------- 聊天接口 -------------------------
def chat(model, tokenizer, prompt, max_new_tokens=200, temperature=0.7):
    model.eval()
    input_text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# 示例
print("问:", "解释量子隧穿效应")
print("答:", chat(model, tokenizer, "解释量子隧穿效应"))

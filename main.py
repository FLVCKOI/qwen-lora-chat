from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

# Load base + LoRA adapter
base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen-1_8B",
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True
)
model = PeftModel.from_pretrained(base_model, "./qwen_lora_adapter")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-1.8B", trust_remote_code=True)

model.eval()

class PromptRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 200
    temperature: float = 0.7

@app.post("/generate")
async def generate_text(req: PromptRequest):
    input_text = f"<|im_start|>user\n{req.prompt}<|im_end|>\n<|im_start|>assistant\n"
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=req.max_new_tokens,
        temperature=req.temperature,
        do_sample=True,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id
    )
    response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
response_only = response_text.split("<|im_start|>assistant")[-1].strip()
return {"response": response_only}

@app.get("/")
def get_index():
    return FileResponse("static/index.html")
1
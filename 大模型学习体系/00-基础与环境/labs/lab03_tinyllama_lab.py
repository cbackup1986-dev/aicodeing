from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

print("TinyLlama实战练习")
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

def generate(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

prompts = [
    "用三句话介绍机器学习。",
    "给出一个三行的中文诗。",
    "解释CUDA的作用。"
]

for p in prompts:
    print(f"\n问题: {p}")
    print(f"回答: {generate(p)}")

try:
    q = input("\n请输入你的问题: ")
    if q.strip():
        print(f"\n回答: {generate(q)}")
except EOFError:
    pass
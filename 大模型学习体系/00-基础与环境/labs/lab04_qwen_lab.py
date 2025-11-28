from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

print("Qwen2.5实战练习")
model_name = "Qwen/Qwen2.5-1.5B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True
)

def chat(prompt, history=None):
    messages = []
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": prompt})
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )
    response = tokenizer.decode(
        outputs[0][len(inputs.input_ids[0]):],
        skip_special_tokens=True
    )
    return response

queries = [
    "列出三个学习深度学习的步骤。",
    "生成一个包含三道题的Python练习。",
    "解释Transformer中的注意力机制。"
]

for q in queries:
    print(f"\n用户: {q}")
    print(f"Qwen: {chat(q)}")

history = []
first = "我想制定一周的学习计划"
print(f"\n用户: {first}")
ans1 = chat(first)
print(f"Qwen: {ans1}")
history.append({"role": "user", "content": first})
history.append({"role": "assistant", "content": ans1})

second = "请根据计划给出每天的任务"
print(f"\n用户: {second}")
ans2 = chat(second, history)
print(f"Qwen: {ans2}")

if torch.cuda.is_available():
    print(f"\n显存使用: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print(f"显存峰值: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
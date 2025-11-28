from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

print("=" * 50)
print("开始加载Qwen2.5模型...")
print("=" * 50)

model_name = "Qwen/Qwen2.5-1.5B-Instruct"

print("\n1. 加载模型和分词器...")
print("(首次运行需要下载模型)")
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
print(f"✓ 加载完成,使用设备: {model.device}")

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

print("\n2. 开始对话测试...")
test_cases = [
    "请详细解释一下什么是大语言模型?",
    "用Python实现一个简单的计算器程序,支持加减乘除。",
    "给我讲一个关于AI的有趣故事。"
]

for i, prompt in enumerate(test_cases, 1):
    print(f"\n{'='*50}")
    print(f"测试 {i}/{len(test_cases)}")
    print(f"{'='*50}")
    print(f"用户: {prompt}")
    response = chat(prompt)
    print(f"Qwen: {response}")

print(f"\n{'='*50}")
print("多轮对话示例")
print(f"{'='*50}")

history = []
prompt1 = "我想学习Python,应该从哪里开始?"
print(f"\n用户: {prompt1}")
response1 = chat(prompt1)
print(f"Qwen: {response1}")
history.append({"role": "user", "content": prompt1})
history.append({"role": "assistant", "content": response1})

prompt2 = "有什么好的学习资源推荐吗?"
print(f"\n用户: {prompt2}")
response2 = chat(prompt2, history)
print(f"Qwen: {response2}")

if torch.cuda.is_available():
    print(f"\n显存使用: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print(f"显存峰值: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
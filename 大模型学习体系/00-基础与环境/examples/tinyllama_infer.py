from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

print("=" * 50)
print("开始加载TinyLlama模型...")
print("=" * 50)

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

print("\n1. 加载分词器...")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
print("✓ 分词器加载完成")

print("\n2. 加载模型(首次运行需要下载)...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)
print(f"✓ 模型加载完成,使用设备: {model.device}")

print("\n3. 开始推理...")
test_prompts = [
    "你好,请介绍一下你自己。",
    "什么是人工智能?",
    "用Python写一个Hello World程序。"
]

for i, prompt in enumerate(test_prompts, 1):
    print(f"\n问题{i}: {prompt}")
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"回答: {response}")

print("\n" + "=" * 50)
print("推理测试完成!")
print("=" * 50)

if torch.cuda.is_available():
    print(f"\n显存使用: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print(f"显存峰值: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
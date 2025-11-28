from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

model_name = "Qwen/Qwen2.5-1.5B-Instruct"
print("加载模型中...")
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
print("✓ 模型加载完成!")

def generate_response(prompt, max_tokens=200):
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        temperature=0.7,
        top_p=0.9,
        do_sample=True
    )
    response = tokenizer.decode(
        outputs[0][len(inputs.input_ids[0]):],
        skip_special_tokens=True
    )
    return response

prompt = "请用一句话解释深度学习"
response = generate_response(prompt)
print(f"问题: {prompt}")
print(f"回答: {response}")

try:
    prompt = input("请输入问题: ")
    if prompt.strip():
        response = generate_response(prompt)
        print(f"\n回答: {response}")
except EOFError:
    pass
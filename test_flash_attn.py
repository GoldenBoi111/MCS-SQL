import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.profiler import profile, ProfilerActivity
import time

# You can change this if you want to test Qwen or another model!
MODEL_NAME = "openai/gpt-oss-120b"

print(f"Loading {MODEL_NAME} in bfloat16...")
# This forces the remote code just like your pipeline does
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    attn_implementation="flash_attention_2"
)

print("\n" + "="*50)
print("=== ARCHITECTURE INSPECTION ===")
print("="*50)
print("Model Class:", type(model))

# Attempt to locate the exact attention layer
try:
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        attn_class = type(model.model.layers[0].self_attn)
        print("Attention Layer Class:", attn_class)
    else:
        print("Custom architecture layout detected. Printing full model to identify attention class:")
        print(model)
except Exception as e:
    print("Error inspecting architecture structure:", e)

print("\n" + "="*50)
print("=== PROFILING CUDA KERNELS ===")
print("="*50)
print("Running a short prompt to observe the mathematical kernels being launched on the GPU...")
prompt = "Explain the importance of database indexing in SQL."
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

# 1. Warm-up (we do this so the profiler doesn't record one-time setup costs)
with torch.no_grad():
    model.generate(**inputs, max_new_tokens=5, pad_token_id=tokenizer.eos_token_id)

print("\n[ Profiling active, generating 20 tokens... ]")
# 2. The actual profiled run
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=20, pad_token_id=tokenizer.eos_token_id)
        
print("\n" + "="*50)
print("=== GENERATED TEXT (CHECK FOR GIBBERISH) ===")
print("="*50)
text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print(text)

# 3. Print the top 15 most expensive operations
print("\n" + "="*50)
print("=== TOP 15 GPU OPERATIONS ===")
print("="*50)
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))

print("\n" + "="*50)
print("=== THE VERDICT ===")
print("="*50)
print("Look closely at the 'Name' column in the table above:")
print("✅ If you see ANY rows containing 'flash_fwd' or 'flash_attn', your model organically found and is successfully using Native Flash Attention 2.")
print("❌ If the top operations are heavily nested generic math like 'bmm', 'softmax', or 'scaled_dot_product', it fell back to Eager PyTorch Attention.")
print("="*50)

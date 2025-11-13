# from huggingface_hub import HfApi, hf_hub_download

# # Step 1: Set the repository ID
# repo_id = "bartowski/DeepSeek-R1-Distill-Qwen-7B-GGUF"

# # Step 2: Create an HfApi instance
# api = HfApi()

# # Step 3: List all files in the repo
# files = api.list_repo_files(repo_id)
# print("Available files in the repo:")
# for f in files:
#     print("-", f)

# # Step 4: Pick a recommended quantized version
# # Options usually: Q4_0, Q4_K_M, Q5_K_M, etc. Choose a smaller quant for CPU
# recommended_quant = "DeepSeek-R1-Distill-Qwen-7B-Q3_K_M.gguf"
# if recommended_quant not in files:
#     raise ValueError(f"{recommended_quant} not found in repo. Available files: {files}")

# # Step 5: Download the file
# file_path = hf_hub_download(repo_id=repo_id, filename=recommended_quant)
# print(f"\n✅ Downloaded model to: {file_path}")

# from gpt4all import GPT4All
# import os

# model_path = r"C:\Users\Admin\Documents\FeedbackAgent\DeepSeek-R1-Distill-Qwen-7B-Q3_K_M.gguf"

# if not os.path.exists(model_path):
#     raise FileNotFoundError(f"Model file not found at: {model_path}")

# try:
#     gpt_model = GPT4All(model_path, model_type="gguf", allow_download=False)

#     # Test if the model is usable
#     try:
#         test = gpt_model.generate("Hello")
#         print("✅ Model is functional!", test)
#     except Exception as gen_error:
#         print("❌ Model loaded object exists, but generation failed!")
#         print("Generation error:", gen_error)

# except Exception as e:
#     print("❌ Failed to load model at all")
#     print(e)
from gpt4all import GPT4All

model_path = "mistral-7b-openorca.Q4_0.gguf"

gpt_model = GPT4All(model_path, allow_download=True)

# Test generation
print(gpt_model.generate("Hello! How are you?"))
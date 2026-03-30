import os

os.environ["HF_HOME"] = "E:/HF_CACHE"
os.environ["HUGGINGFACE_HUB_CACHE"] = "E:/HF_CACHE/hub"
os.environ["TRANSFORMERS_CACHE"] = "E:/HF_CACHE/transformers"


from transformers import AutoTokenizer, AutoModel

model_name = "ProsusAI/finbert"

save_path = "E:/FinBERT/ProsusAI-finbert"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

tokenizer.save_pretrained(save_path)
model.save_pretrained(save_path)

print("Downloaded successfully!")
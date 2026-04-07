import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F

# ===== CONFIG =====
MODEL_PATH = "E:/FinBERT/ProsusAI-finbert"

# Label mapping for FinBERT
LABELS = ["negative", "neutral", "positive"]

# ===== LOAD MODEL =====
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

print(f"Model loaded on: {device}")

# ===== CORE FUNCTION =====
def predict_sentiment(texts):
    """
    texts: list of strings
    returns: list of dicts with sentiment + confidence
    """

    # Edge case: empty input
    if not texts:
        return []

    # Clean input
    texts = [t if isinstance(t, str) else "" for t in texts]

    # Tokenize
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Inference
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    probs = F.softmax(logits, dim=1)

    results = []
    for i, prob in enumerate(probs):
        confidence, predicted_class = torch.max(prob, dim=0)

        results.append({
            "text": texts[i],
            "sentiment": LABELS[predicted_class.item()],
            "confidence": round(confidence.item(), 4),
            "probabilities": {
                LABELS[j]: round(prob[j].item(), 4)
                for j in range(len(LABELS))
            }
        })

    return results


# ===== TEST CASES =====
if __name__ == "__main__":
    test_texts = [
        "Reliance stock surged after strong quarterly earnings.",
        "The company reported a massive loss and declining revenue.",
        "Market remained flat with mixed investor sentiment.",
        "",  # edge case
        "Adani shares crash badly after fraud allegations and panic selling"
    ]

    results = predict_sentiment(test_texts)

    print("\n===== RESULTS =====\n")
    for res in results:
        print(f"Text: {res['text']}")
        print(f"Sentiment: {res['sentiment']} | Confidence: {res['confidence']}")
        print(f"Probabilities: {res['probabilities']}")
        print("-" * 60)
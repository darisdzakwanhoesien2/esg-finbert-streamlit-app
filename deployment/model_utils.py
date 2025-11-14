# deployment/model_utils.py

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from huggingface_hub import hf_hub_download

SENTIMENT_LABELS = {0: "positive", 1: "neutral", 2: "negative", 3: "none"}
TONE_LABELS = {0: "Action", 1: "Commitment", 2: "Outcome"}

class MultiTaskFinBERT(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        hidden = encoder.config.hidden_size
        self.encoder = encoder
        self.dropout = nn.Dropout(0.2)
        self.sentiment_head = nn.Linear(hidden, 4)
        self.tone_head = nn.Linear(hidden, 3)

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids, attention_mask=attention_mask)
        pooled = self.dropout(out.last_hidden_state[:, 0])
        return self.sentiment_head(pooled), self.tone_head(pooled)

def load_model_from_hub(repo_id):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(repo_id)

    encoder = AutoModel.from_pretrained(repo_id)
    model = MultiTaskFinBERT(encoder)

    # download weights
    model_path = hf_hub_download(repo_id, "pytorch_model.bin")
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()

    return model, tokenizer, device

def predict_texts(model, tokenizer, texts, device="cpu"):
    if isinstance(texts, str):
        texts = [texts]

    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=256,
        return_tensors="pt"
    )

    input_ids = encoded["input_ids"].to(device)
    mask = encoded["attention_mask"].to(device)

    with torch.no_grad():
        sent_log, tone_log = model(input_ids, mask)
        sent_prob = torch.softmax(sent_log, dim=1).cpu().numpy()
        tone_prob = torch.softmax(tone_log, dim=1).cpu().numpy()

    results = []
    for i, t in enumerate(texts):
        results.append({
            "text": t,
            "sentiment": SENTIMENT_LABELS[sent_prob[i].argmax()],
            "tone": TONE_LABELS[tone_prob[i].argmax()],
            "sentiment_scores": sent_prob[i].tolist(),
            "tone_scores": tone_prob[i].tolist()
        })

    return results

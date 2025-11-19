import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from huggingface_hub import hf_hub_download, list_repo_files, HfApi

SENTIMENT_LABELS = {0: "positive", 1: "neutral", 2: "negative", 3: "none"}
TONE_LABELS = {0: "Action", 1: "Commitment", 2: "Outcome"}


# ------------------------------------------------------
# MultiTask Model Definition
# ------------------------------------------------------
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


# ------------------------------------------------------
# Intelligent file-selection for HF model weights
# ------------------------------------------------------
def find_checkpoint_file(repo_id):
    """
    Detect automatically which model weight file exists.
    Returns the filename that should be downloaded.
    """
    files = list_repo_files(repo_id)

    # Priority order for most common file names
    candidates = [
        "pytorch_model.bin",
        "model.safetensors",
        "decoder_model.safetensors",
        "model.ckpt",
        "tf_model.h5",
    ]

    # Direct match
    for c in candidates:
        if c in files:
            return c

    # Search for weights inside subfolders (checkpoint directories)
    for f in files:
        if any(f.endswith(ext) for ext in (".bin", ".safetensors", ".ckpt")):
            return f  # return first found

    raise FileNotFoundError(
        f"No model weight file found in repo '{repo_id}'. "
        f"Available files: {files}"
    )


# ------------------------------------------------------
# Load model from HuggingFace Hub (Robust)
# ------------------------------------------------------
def load_model_from_hub(repo_id):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load tokenizer + base encoder
    tokenizer = AutoTokenizer.from_pretrained(repo_id)
    encoder = AutoModel.from_pretrained(repo_id)

    model = MultiTaskFinBERT(encoder)

    # Find correct weight file
    weight_file = find_checkpoint_file(repo_id)
    print(f"[INFO] Using checkpoint file: {weight_file}")

    # Download from HF
    model_path = hf_hub_download(repo_id, weight_file)

    # Load weights
    if weight_file.endswith(".safetensors"):
        from safetensors.torch import load_file
        state = load_file(model_path)
    else:
        state = torch.load(model_path, map_location=device)

    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()

    return model, tokenizer, device


# ------------------------------------------------------
# Prediction helper
# ------------------------------------------------------
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
        sent_logits, tone_logits = model(input_ids, mask)
        sent_probs = torch.softmax(sent_logits, dim=1).cpu().numpy()
        tone_probs = torch.softmax(tone_logits, dim=1).cpu().numpy()

    results = []
    for i, txt in enumerate(texts):
        results.append({
            "text": txt,
            "sentiment": SENTIMENT_LABELS[sent_probs[i].argmax()],
            "tone": TONE_LABELS[tone_probs[i].argmax()],
            "sentiment_scores": sent_probs[i].tolist(),
            "tone_scores": tone_probs[i].tolist(),
        })

    return results

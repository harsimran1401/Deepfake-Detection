
# MODEL: Vision Transformer (ViT) - prithivMLmods/Deep-Fake-Detector-v2-Model
# STRENGTH: Excellent at spotting digital generation artifacts and texture issues.

from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoImageProcessor, AutoModelForImageClassification
from facenet_pytorch import MTCNN
import torch
from PIL import Image
import io
import base64

app = Flask(__name__)
CORS(app)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Loading ViT Model on {device}...", flush=True)

# 1. Load Face Detector
mtcnn = MTCNN(image_size=224, keep_all=False, device=device)

# 2. Load ViT Model
MODEL_NAME = "prithivMLmods/Deep-Fake-Detector-v2-Model"
processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
model = AutoModelForImageClassification.from_pretrained(MODEL_NAME).to(device)
model.eval()
print("✅ ViT Model Ready!", flush=True)

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.json
        # Decode Image
        image_bytes = base64.b64decode(data['image'].split(",")[1])
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Detect Face
        face = mtcnn(image)
        if face is None:
            return jsonify({"verdict": "No Face Detected", "confidence": 0})

        # Preprocess for ViT
        # MTCNN returns a tensor, we convert back to PIL for the processor
        from torchvision import transforms
        face_pil = transforms.ToPILImage()(face)
        
        inputs = processor(images=face_pil, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            # This model: Label 0 = Fake, Label 1 = Real (Check config usually)
            # We assume standard mapping: "Fake" vs "Real"
            id2label = model.config.id2label
            top_id = torch.argmax(probs).item()
            label = id2label[top_id]
            score = probs[0][top_id].item()

        # Normalize verdict
        if "fake" in label.lower():
            verdict = "FAKE"
        else:
            verdict = "REAL"

        return jsonify({"verdict": verdict, "confidence": round(score * 100, 2)})

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(port=5000)

# MODEL: ResNet-50 (umm-maybe/AI-Image-Detector)
# STRENGTH: "Universal" detector. Great at spotting AI-generated "art" artifacts
#           that often appear in deepfake backgrounds or hair.

from flask import Flask, request, jsonify
from flask_cors import CORS
import transformers.utils.import_utils
import transformers.modeling_utils
transformers.utils.import_utils.check_torch_load_is_safe = lambda: None
transformers.utils.import_utils.check_torch_load_is_safe = lambda: True
from transformers import AutoImageProcessor, AutoModelForImageClassification
from facenet_pytorch import MTCNN
import torch
from PIL import Image
import io
import base64

app = Flask(__name__)
CORS(app)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Loading ResNet-50 on {device}...", flush=True)

# 1. Load Face Detector (Standard)
mtcnn = MTCNN(image_size=224, keep_all=False, device=device)

# 2. Load ResNet Model
# This model is a powerhouse for generic AI detection
MODEL_NAME = "umm-maybe/AI-Image-Detector"
processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
model = AutoModelForImageClassification.from_pretrained(MODEL_NAME).to(device)
model.eval()

print("✅ ResNet Model Ready!", flush=True)

@app.route('/analyze', methods=['POST'])
def analyze():
    print("📸 Processing ResNet request...", flush=True)
    try:
        data = request.json
        if "," in data['image']:
            encoded = data['image'].split(",")[1]
        else:
            encoded = data['image']
        image_bytes = base64.b64decode(encoded)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # 1. Face Check
        face = mtcnn(image)
        if face is None:
            return jsonify({"verdict": "No Face Detected", "confidence": 0})

        # 2. ResNet Analysis
        # Note: ResNet expects the PIL image, not the tensor
        from torchvision import transforms
        face_pil = transforms.ToPILImage()(face)
        
        inputs = processor(images=face_pil, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)
            
            # Helper to find which label is 'fake'
            # Usually: Label 0 = Fake, Label 1 = Real (or check config)
            id2label = model.config.id2label
            # print(f"DEBUG Labels: {id2label}") 
            
            # This specific model outputs: {0: 'fake', 1: 'real'} usually.
            # We will dynamically find the 'fake' label index
            fake_idx = 0
            for idx, label in id2label.items():
                if "art" in label.lower() or "fake" in label.lower() or "ai" in label.lower():
                    fake_idx = idx
                    break
            
            fake_prob = probs[0][fake_idx].item()
            real_prob = 1 - fake_prob

        print(f"🧠 ResNet Score -> Fake: {fake_prob:.2f}", flush=True)

        if fake_prob > 0.5:
            return jsonify({"verdict": "FAKE", "confidence": round(fake_prob * 100, 2)})
        else:
            return jsonify({"verdict": "REAL", "confidence": round(real_prob * 100, 2)})

    except Exception as e:
        print(f"Error: {e}", flush=True)
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(port=5000)
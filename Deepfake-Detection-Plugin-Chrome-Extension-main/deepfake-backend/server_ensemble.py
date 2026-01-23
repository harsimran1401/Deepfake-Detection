
# STRATEGY: ENSEMBLE (ViT + EfficientNet)
# RESULT: Averaged probability for higher stability.

from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoImageProcessor, AutoModelForImageClassification
from torchvision import models, transforms
from facenet_pytorch import MTCNN
import torch
import torch.nn as nn
from PIL import Image
import io
import base64

app = Flask(__name__)
CORS(app)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Loading ENSEMBLE Models on {device}...", flush=True)

# --- LOAD JOINT RESOURCES ---
mtcnn = MTCNN(image_size=224, keep_all=False, device=device)

# --- LOAD MODEL 1: ViT ---
print("1️⃣ Loading ViT...", flush=True)
vit_name = "prithivMLmods/Deep-Fake-Detector-v2-Model"
vit_processor = AutoImageProcessor.from_pretrained(vit_name)
vit_model = AutoModelForImageClassification.from_pretrained(vit_name).to(device)
vit_model.eval()

# --- LOAD MODEL 2: EfficientNet ---
print("2️⃣ Loading EfficientNet...", flush=True)
eff_model = models.efficientnet_b0(weights=None)
eff_model.classifier[1] = nn.Linear(eff_model.classifier[1].in_features, 2)
state_dict = torch.hub.load_state_dict_from_url(
    "https://huggingface.co/Xicor9/efficientnet-b0-ffpp-c23/resolve/main/efficientnet_b0_ffpp_c23.pth",
    map_location=device
)
eff_model.load_state_dict(state_dict)
eff_model.to(device)
eff_model.eval()

# Helper for EfficientNet Preprocessing
eff_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

print("✅ ALL SYSTEMS GO!", flush=True)

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        # Get Image
        data = request.json
        image_bytes = base64.b64decode(data['image'].split(",")[1])
        original_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # 1. Detect Face (Shared)
        face_tensor = mtcnn(original_image)
        if face_tensor is None:
            return jsonify({"verdict": "No Face Detected", "confidence": 0})
        
        # Convert tensor back to PIL for individual preprocessing
        from torchvision import transforms as T
        face_pil = T.ToPILImage()(face_tensor)

        # 2. Get Score from ViT
        inputs = vit_processor(images=face_pil, return_tensors="pt").to(device)
        with torch.no_grad():
            vit_out = vit_model(**inputs)
            vit_probs = torch.nn.functional.softmax(vit_out.logits, dim=-1)
            # Assuming Label 0=Fake, 1=Real for ViT (Check your specific model config!)
            # Note: prithivMLmods usually maps 0: Fake, 1: Real. 
            # We want the "Fake" probability.
            vit_fake_prob = vit_probs[0][0].item() if "fake" in vit_model.config.id2label[0].lower() else vit_probs[0][1].item()

        # 3. Get Score from EfficientNet
        eff_input = eff_transform(face_pil).unsqueeze(0).to(device)
        with torch.no_grad():
            eff_out = eff_model(eff_input)
            eff_probs = torch.nn.functional.softmax(eff_out, dim=1)
            # FFPP models usually: 0=Real, 1=Fake.
            eff_fake_prob = eff_probs[0][1].item()

        # 4. The "Verdict" (Average)
        avg_fake_prob = (vit_fake_prob + eff_fake_prob) / 2
        
        print(f"📊 ViT: {vit_fake_prob:.2f} | EffNet: {eff_fake_prob:.2f} | AVG: {avg_fake_prob:.2f}")

        if avg_fake_prob > 0.5:
            return jsonify({"verdict": "FAKE", "confidence": round(avg_fake_prob * 100, 2)})
        else:
            return jsonify({"verdict": "REAL", "confidence": round((1 - avg_fake_prob) * 100, 2)})

    except Exception as e:
        print(e)
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(port=5000)
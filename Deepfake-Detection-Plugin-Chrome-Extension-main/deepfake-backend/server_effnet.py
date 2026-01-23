

# MODEL: EfficientNet-B0 - Trained on FaceForensics++ (FFPP)
# STRENGTH: Great at spotting "Face Swaps" and video compression artifacts.

from flask import Flask, request, jsonify
from flask_cors import CORS
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
print(f"🚀 Loading EfficientNet on {device}...", flush=True)

# 1. Load Face Detector
mtcnn = MTCNN(image_size=224, keep_all=False, device=device)

# 2. Load EfficientNet (Specialized Weights)
model = models.efficientnet_b0(weights=None)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)

try:
    # Load weights specifically trained on deepfake datasets
    state_dict = torch.hub.load_state_dict_from_url(
        "https://huggingface.co/Xicor9/efficientnet-b0-ffpp-c23/resolve/main/efficientnet_b0_ffpp_c23.pth",
        map_location=device
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print("✅ EfficientNet Ready!", flush=True)
except Exception as e:
    print("❌ Error loading weights. Check internet.", e)

# Preprocessing is critical for EfficientNet
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.json
        image_bytes = base64.b64decode(data['image'].split(",")[1])
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        face = mtcnn(image)
        if face is None:
            return jsonify({"verdict": "No Face Detected", "confidence": 0})

        # Prepare face for model
        face_tensor = transform(transforms.ToPILImage()(face)).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(face_tensor)
            probs = torch.nn.functional.softmax(output, dim=1)
            # 0 = Real, 1 = Fake (Specific to this model weight)
            fake_prob = probs[0][1].item()
        
        if fake_prob > 0.5:
            return jsonify({"verdict": "FAKE", "confidence": round(fake_prob * 100, 2)})
        else:
            return jsonify({"verdict": "REAL", "confidence": round((1 - fake_prob) * 100, 2)})

    except Exception as e:
        print(e)
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(port=5000)

# MODEL: OpenAI CLIP (Zero-Shot Detection)
# STRATEGY: We ask the model "Is this a real face?" vs "Is this a deepfake?"

from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import CLIPProcessor, CLIPModel
from facenet_pytorch import MTCNN
import torch
from PIL import Image
import io
import base64

app = Flask(__name__)
CORS(app)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Loading CLIP on {device}...", flush=True)

mtcnn = MTCNN(image_size=224, keep_all=False, device=device)

# Load CLIP (using the robust generic model)
model_name = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_name, use_safetensors=True).to(device)
processor = CLIPProcessor.from_pretrained(model_name)
print("✅ CLIP Ready!", flush=True)

@app.route('/analyze', methods=['POST'])
def analyze():
    print("📸 Processing CLIP request...", flush=True)
    try:
        data = request.json
        if "," in data['image']:
            encoded = data['image'].split(",")[1]
        else:
            encoded = data['image']
        image_bytes = base64.b64decode(encoded)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        face = mtcnn(image)
        if face is None:
            return jsonify({"verdict": "No Face Detected", "confidence": 0})

        # We need PIL for CLIP
        from torchvision import transforms
        face_pil = transforms.ToPILImage()(face)

        # ZERO-SHOT PROMPTING
        # We tell CLIP to compare the image against these two text descriptions
        labels = ["a photo of a real human face", "a deepfake or ai generated face"]
        
        inputs = processor(text=labels, images=face_pil, return_tensors="pt", padding=True).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            # Logits per image are the similarity scores
            logits_per_image = outputs.logits_per_image 
            probs = logits_per_image.softmax(dim=1) 
            
            real_score = probs[0][0].item()
            fake_score = probs[0][1].item()

        print(f"🧠 CLIP Scores -> Real: {real_score:.2f}, Fake: {fake_score:.2f}", flush=True)

        if fake_score > real_score:
            return jsonify({"verdict": "FAKE", "confidence": round(fake_score * 100, 2)})
        else:
            return jsonify({"verdict": "REAL", "confidence": round(real_score * 100, 2)})

    except Exception as e:
        print(f"Error: {e}", flush=True)
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(port=5000)
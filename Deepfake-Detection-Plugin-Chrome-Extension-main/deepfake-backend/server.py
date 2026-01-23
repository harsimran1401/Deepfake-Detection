
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
from facenet_pytorch import MTCNN
import torch
import io
import base64
import sys

app = Flask(__name__)
CORS(app)

print("--------------------------------------------------")
print("🚀 STARTING SERVER... PLEASE WAIT")
print("--------------------------------------------------", flush=True)

# 1. SETUP DEVICE
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"⚙️  Using Device: {device}", flush=True)

# 2. LOAD MODELS
try:
    print("⏳ Loading Face Detector (MTCNN)...", flush=True)
    mtcnn = MTCNN(image_size=224, keep_all=False, select_largest=True, device=device)
    
    print("⏳ Loading AI Model (ViT)... (This takes ~30 seconds)", flush=True)
    MODEL_NAME = "prithivMLmods/Deep-Fake-Detector-v2-Model"
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    model = AutoModelForImageClassification.from_pretrained(MODEL_NAME)
    model.to(device)
    model.eval()
    print("✅ MODELS LOADED SUCCESSFULLY!", flush=True)
except Exception as e:
    print(f"\n❌ CRITICAL ERROR LOADING MODELS: {e}")
    print("👉 HINT: Did you run 'pip install scikit-learn timm'?")
    sys.exit(1) # Stop the server if models fail

# 3. ROOT ROUTE (Browser Health Check)
@app.route('/')
def health_check():
    return "<h1>✅ Deepfake Server is RUNNING!</h1>"

# 4. ANALYZE ROUTE
@app.route('/analyze', methods=['POST'])
def analyze_frame():
    print("📸 Received a frame...", flush=True)
    try:
        data = request.json
        image_data = data.get('image')

        if not image_data:
            return jsonify({"error": "No image"}), 400

        # Decode
        if "," in image_data: header, encoded = image_data.split(",", 1)
        else: encoded = image_data
        image_bytes = base64.b64decode(encoded)
        full_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Detect Face
        face_tensor = mtcnn(full_image)
        if face_tensor is None:
            print("🤷 No face detected", flush=True)
            return jsonify({"verdict": "No Face Detected", "confidence": 0})

        # Predict
        from torchvision import transforms
        face_pil = transforms.ToPILImage()(face_tensor)
        
        inputs = processor(images=face_pil, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            # This specific model maps: 0 -> Fake, 1 -> Real (or vice versa, checking config)
            # Usually id2label tells us. 
            id2label = model.config.id2label
            top_id = torch.argmax(probs).item()
            label = id2label[top_id]
            score = probs[0][top_id].item()

        print(f"🧠 AI Verdict: {label} ({round(score*100, 1)}%)", flush=True)

        # Normalize for Extension (Extension expects "REAL" or "FAKE")
        if "fake" in label.lower():
            verdict = "FAKE"
        else:
            verdict = "REAL"

        return jsonify({"verdict": verdict, "confidence": round(score*100, 2)})

    except Exception as e:
        print(f"❌ Processing Error: {e}", flush=True)
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("🟢 Server starting on Port 5000...", flush=True)
    app.run(debug=True, port=5000)
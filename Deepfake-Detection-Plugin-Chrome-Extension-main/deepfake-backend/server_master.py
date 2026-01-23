
# Version 1 - Ensemble Server
# Combines: 
# 1. EfficientNet-B0 (FaceForensics++)
# 2. ViT (Vision Transformer)
# 3. OpenAI CLIP (Zero-Shot)
# 4. OpenCV Heuristic (Blur/Quality Check)

import warnings
warnings.filterwarnings("ignore") 

from flask import Flask, request, jsonify
from flask_cors import CORS
from facenet_pytorch import MTCNN
import torch
import torch.nn as nn
from torchvision import models, transforms
from transformers import AutoImageProcessor, AutoModelForImageClassification, CLIPProcessor, CLIPModel
from PIL import Image
import io
import base64
import time
import cv2          
import numpy as np  

app = Flask(__name__)
CORS(app)

# -----------------------------------------------------------------------------
# 1. SETUP & LOADING
# -----------------------------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\n🚀 STARTING MASTER SERVER ON [{device.upper()}]...", flush=True)

# --- A. SHARED FACE DETECTOR ---
print("   [1/5] Loading MTCNN...", flush=True)
try:
    mtcnn = MTCNN(image_size=224, keep_all=False, select_largest=True, device=device)
    print("      ✅ MTCNN Ready")
except Exception as e:
    print(f"      ❌ MTCNN Failed: {e}")

# --- B. MODEL 1: ViT ---
print("   [2/5] Loading ViT...", flush=True)
try:
    vit_name = "prithivMLmods/Deep-Fake-Detector-v2-Model"
    vit_processor = AutoImageProcessor.from_pretrained(vit_name)
    vit_model = AutoModelForImageClassification.from_pretrained(vit_name).to(device)
    vit_model.eval()
    print("      ✅ ViT Loaded")
except Exception as e:
    print(f"      ❌ ViT Failed: {e}")
    vit_model = None

# --- C. MODEL 2: EfficientNet ---
print("   [3/5] Loading EfficientNet...", flush=True)
try:
    eff_model = models.efficientnet_b0(weights=None)
    eff_model.classifier[1] = nn.Linear(eff_model.classifier[1].in_features, 2)
    state_dict = torch.hub.load_state_dict_from_url(
        "https://huggingface.co/Xicor9/efficientnet-b0-ffpp-c23/resolve/main/efficientnet_b0_ffpp_c23.pth",
        map_location=device
    )
    eff_model.load_state_dict(state_dict)
    eff_model.to(device)
    eff_model.eval()
    eff_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    print("      ✅ EfficientNet Loaded")
except Exception as e:
    print(f"      ❌ EfficientNet Failed: {e}")
    eff_model = None

# --- D. MODEL 3: OpenAI CLIP ---
print("   [4/5] Loading OpenAI CLIP...", flush=True)
try:
    clip_name = "openai/clip-vit-base-patch32"
    clip_model = CLIPModel.from_pretrained(clip_name, use_safetensors=True).to(device)
    clip_processor = CLIPProcessor.from_pretrained(clip_name)
    print("      ✅ CLIP Loaded")
except Exception as e:
    print(f"      ❌ CLIP Failed: {e}")
    clip_model = None

# --- E. MODEL 4: Simple Model ---
print("   [5/5] Loading Simple Model (OpenCV)...", flush=True)
# It's just a library import, but let's make it look consistent in the logs!
print("      ✅ Simple Model Ready")

print("\n🟢 SERVER READY! Waiting for browser extension...\n")

# -----------------------------------------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------------------------------------
def get_vit_prediction(face_pil):
    if vit_model is None: return 0.5
    try:
        inputs = vit_processor(images=face_pil, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = vit_model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            fake_score = 0.5
            for i, label in vit_model.config.id2label.items():
                if "fake" in label.lower():
                    fake_score = probs[0][i].item()
                    break
            else:
                fake_score = probs[0][0].item() 
        return fake_score
    except: return 0.5

def get_effnet_prediction(face_pil):
    if eff_model is None: return 0.5
    try:
        img_tensor = eff_transform(face_pil).unsqueeze(0).to(device)
        with torch.no_grad():
            output = eff_model(img_tensor)
            probs = torch.nn.functional.softmax(output, dim=1)
            return probs[0][1].item() 
    except: return 0.5

def get_clip_prediction(face_pil):
    if clip_model is None: return 0.5
    try:
        prompts = ["a photo of a real human face", "a deepfake generated face"]
        inputs = clip_processor(text=prompts, images=face_pil, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            outputs = clip_model(**inputs)
            probs = outputs.logits_per_image.softmax(dim=1)
            return probs[0][1].item()
    except: return 0.5

def get_simple_prediction(face_pil):
    try:
        img_cv = np.array(face_pil)
        gray = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)
        variance = cv2.Laplacian(gray, cv2.CV_64F).var()
        if variance < 100: return 0.8
        else: return 0.2
    except: return 0.5

# -----------------------------------------------------------------------------
# MAIN ANALYZE ROUTE
# -----------------------------------------------------------------------------
@app.route('/analyze', methods=['POST'])
def analyze():
    start_time = time.time()
    try:
        # 1. Decode Image
        data = request.json
        if "," in data['image']: encoded = data['image'].split(",")[1]
        else: encoded = data['image']
        image_bytes = base64.b64decode(encoded)
        original_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # 2. Detect Face
        face_tensor = mtcnn(original_image)
        if face_tensor is None:
            return jsonify({"verdict": "No Face Detected", "confidence": 0})
        
        face_pil = transforms.ToPILImage()(face_tensor)

        # 3. RUN ALL 4 MODELS
        score_eff = get_effnet_prediction(face_pil)
        score_vit = get_vit_prediction(face_pil)
        score_clip = get_clip_prediction(face_pil)
        score_simple = get_simple_prediction(face_pil)

        # 4. ENSEMBLE LOGIC
        w_eff = 0.35    
        w_vit = 0.30    
        w_clip = 0.20   
        w_simple = 0.15 
        
        final_score = (score_eff * w_eff) + (score_vit * w_vit) + (score_clip * w_clip) + (score_simple * w_simple)
        
        verdict = "FAKE" if final_score > 0.5 else "REAL"
        confidence = round(final_score * 100, 2) if verdict == "FAKE" else round((1 - final_score) * 100, 2)
        process_time = round(time.time() - start_time, 3)

        # 5. TERMINAL DASHBOARD
        print("\n" + "="*65)
        print(f"📸 FRAME ANALYZED ({process_time}s)")
        print("-" * 65)
        print(f" {'MODEL':<18} | {'PROB (FAKE)':<12} | {'VERDICT':<10}")
        print("-" * 65)
        print(f" {'EfficientNet':<18} | {score_eff:.4f}       | {'🟥 FAKE' if score_eff > 0.5 else '🟩 REAL'}")
        print(f" {'ViT-Base':<18} | {score_vit:.4f}       | {'🟥 FAKE' if score_vit > 0.5 else '🟩 REAL'}")
        print(f" {'OpenAI CLIP':<18} | {score_clip:.4f}       | {'🟥 FAKE' if score_clip > 0.5 else '🟩 REAL'}")
        print(f" {'Simple (OpenCV)':<18} | {score_simple:.4f}       | {'🟥 FAKE' if score_simple > 0.5 else '🟩 REAL'}")
        print("-" * 65)
        print(f" 🏆 ENSEMBLE RESULT:  {final_score:.4f} -> {verdict} ({confidence}%)")
        print("="*65 + "\n")

        # return jsonify({"verdict": verdict, "confidence": confidence})
        ##new addition below
        return jsonify({
            "verdict": verdict, 
            "confidence": confidence,
            "details": {
                "EfficientNet": score_eff,
                "ViT": score_vit,
                "OpenAI CLIP": score_clip,
                "Simple (OpenCV)": score_simple
            }
        })

    except Exception as e:
        print(f"❌ Error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(port=5000, debug=True)


#Modifieid Algorithmic Predictions - POORER ACCURACY IN MY OPINION
# import warnings
# warnings.filterwarnings("ignore") 

# from flask import Flask, request, jsonify
# from flask_cors import CORS
# from facenet_pytorch import MTCNN
# import torch
# import torch.nn as nn
# from torchvision import models, transforms
# from transformers import AutoImageProcessor, AutoModelForImageClassification, CLIPProcessor, CLIPModel
# from PIL import Image
# import io
# import base64
# import time
# import cv2          
# import numpy as np  

# app = Flask(__name__)
# CORS(app)

# # -----------------------------------------------------------------------------
# # 1. SETUP & LOADING
# # -----------------------------------------------------------------------------
# device = "cuda" if torch.cuda.is_available() else "cpu"
# print(f"\n🚀 STARTING MASTER SERVER ON [{device.upper()}]...", flush=True)

# # --- A. SHARED FACE DETECTOR ---
# print("   [1/5] Loading MTCNN...", flush=True)
# try:
#     # Keep exact settings from your working version
#     mtcnn = MTCNN(image_size=224, keep_all=False, select_largest=True, device=device)
#     print("      ✅ MTCNN Ready")
# except Exception as e:
#     print(f"      ❌ MTCNN Failed: {e}")

# # --- B. MODEL 1: ViT ---
# print("   [2/5] Loading ViT...", flush=True)
# try:
#     vit_name = "prithivMLmods/Deep-Fake-Detector-v2-Model"
#     vit_processor = AutoImageProcessor.from_pretrained(vit_name)
#     vit_model = AutoModelForImageClassification.from_pretrained(vit_name).to(device)
#     vit_model.eval()
#     print("      ✅ ViT Loaded")
# except Exception as e:
#     print(f"      ❌ ViT Failed: {e}")
#     vit_model = None

# # --- C. MODEL 2: EfficientNet ---
# print("   [3/5] Loading EfficientNet...", flush=True)
# try:
#     eff_model = models.efficientnet_b0(weights=None)
#     eff_model.classifier[1] = nn.Linear(eff_model.classifier[1].in_features, 2)
#     state_dict = torch.hub.load_state_dict_from_url(
#         "https://huggingface.co/Xicor9/efficientnet-b0-ffpp-c23/resolve/main/efficientnet_b0_ffpp_c23.pth",
#         map_location=device
#     )
#     eff_model.load_state_dict(state_dict)
#     eff_model.to(device)
#     eff_model.eval()
#     eff_transform = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ])
#     print("      ✅ EfficientNet Loaded")
# except Exception as e:
#     print(f"      ❌ EfficientNet Failed: {e}")
#     eff_model = None

# # --- D. MODEL 3: OpenAI CLIP ---
# print("   [4/5] Loading OpenAI CLIP...", flush=True)
# try:
#     clip_name = "openai/clip-vit-base-patch32"
#     clip_model = CLIPModel.from_pretrained(clip_name, use_safetensors=True).to(device)
#     clip_processor = CLIPProcessor.from_pretrained(clip_name)
#     print("      ✅ CLIP Loaded")
# except Exception as e:
#     print(f"      ❌ CLIP Failed: {e}")
#     clip_model = None

# # --- E. MODEL 4: Simple Model ---
# print("   [5/5] Loading Simple Model (OpenCV)...", flush=True)
# print("      ✅ Simple Model Ready")

# print("\n🟢 SERVER READY! Waiting for browser extension...\n")

# # -----------------------------------------------------------------------------
# # HELPER FUNCTIONS
# # -----------------------------------------------------------------------------

# # ✅ NEW: Specialized Robust Function for EfficientNet
# def get_effnet_prediction_robust(original_image):
#     if eff_model is None: return 0.5
#     try:
#         # 1. Detect coordinates (Does NOT alter colors)
#         boxes, _ = mtcnn.detect(original_image)
#         if boxes is None: return 0.5

#         # 2. Convert coordinates to Integers (Crucial Fix!)
#         # The crash likely happened here because crop() hates decimals.
#         box = [int(b) for b in boxes[0]]
        
#         # 3. Crop manually
#         face_crop = original_image.crop((box[0], box[1], box[2], box[3]))
        
#         # 4. Predict
#         img_tensor = eff_transform(face_crop).unsqueeze(0).to(device)
#         with torch.no_grad():
#             output = eff_model(img_tensor)
#             probs = torch.nn.functional.softmax(output, dim=1)
#             return probs[0][1].item()
#     except Exception as e: 
#         print(f"   ⚠️ EffNet Robust Error: {e}") # Print error to terminal if it fails
#         return 0.5

# def get_vit_prediction(face_pil):
#     if vit_model is None: return 0.5
#     try:
#         inputs = vit_processor(images=face_pil, return_tensors="pt").to(device)
#         with torch.no_grad():
#             outputs = vit_model(**inputs)
#             probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
#             fake_score = 0.5
#             for i, label in vit_model.config.id2label.items():
#                 if "fake" in label.lower():
#                     fake_score = probs[0][i].item()
#                     break
#             else:
#                 fake_score = probs[0][0].item() 
#         return fake_score
#     except: return 0.5

# def get_clip_prediction(face_pil):
#     if clip_model is None: return 0.5
#     try:
#         prompts = ["a photo of a real human face", "a deepfake generated face"]
#         inputs = clip_processor(text=prompts, images=face_pil, return_tensors="pt", padding=True).to(device)
#         with torch.no_grad():
#             outputs = clip_model(**inputs)
#             probs = outputs.logits_per_image.softmax(dim=1)
#             return probs[0][1].item()
#     except: return 0.5

# def get_simple_prediction(face_pil):
#     try:
#         img_cv = np.array(face_pil)
#         gray = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)
#         variance = cv2.Laplacian(gray, cv2.CV_64F).var()
#         if variance < 100: return 0.8
#         else: return 0.2
#     except: return 0.5

# # -----------------------------------------------------------------------------
# # MAIN ANALYZE ROUTE
# # -----------------------------------------------------------------------------
# @app.route('/analyze', methods=['POST'])
# def analyze():
#     start_time = time.time()
#     try:
#         # 1. Decode Image
#         data = request.json
#         if "," in data['image']: encoded = data['image'].split(",")[1]
#         else: encoded = data['image']
#         image_bytes = base64.b64decode(encoded)
#         original_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

#         # 2. GLOBAL DETECTION (Your Working Code)
#         # We assume this works for everyone else
#         face_tensor = mtcnn(original_image)
#         if face_tensor is None:
#             return jsonify({"verdict": "No Face Detected", "confidence": 0})
        
#         face_pil_general = transforms.ToPILImage()(face_tensor)

#         # 3. RUN MODELS
#         # ✅ A. Use Robust EffNet (Passes original image)
#         score_eff = get_effnet_prediction_robust(original_image)
        
#         # ✅ B. Use General for others
#         score_vit = get_vit_prediction(face_pil_general)
#         score_clip = get_clip_prediction(face_pil_general)
#         score_simple = get_simple_prediction(face_pil_general)

#         # 4. ENSEMBLE LOGIC
#         w_eff = 0.25   
#         w_vit = 0.25    
#         w_clip = 0.25 
#         w_simple = 0.25 
        
#         final_score = (score_eff * w_eff) + (score_vit * w_vit) + (score_clip * w_clip) + (score_simple * w_simple)
        
#         verdict = "FAKE" if final_score > 0.5 else "REAL"
#         # confidence = [round(final_score * 100, 2) if verdict == "FAKE" else round((1 - final_score) * 100, 2), f'\n{score_eff}', f'\n{score_vit}', score_clip, score_simple]
#         confidence = round(final_score * 100, 2) if verdict == "FAKE" else round((1 - final_score) * 100, 2)
#         process_time = round(time.time() - start_time, 3)

#         # 5. TERMINAL DASHBOARD
#         print("\n" + "="*65)
#         print(f"📸 FRAME ANALYZED ({process_time}s)")
#         print("-" * 65)
#         print(f" {'MODEL':<18} | {'PROB (FAKE)':<12} | {'VERDICT':<10}")
#         print("-" * 65)
#         print(f" {'EfficientNet':<18} | {score_eff:.4f}       | {'🟥 FAKE' if score_eff > 0.5 else '🟩 REAL'}")
#         print(f" {'ViT-Base':<18} | {score_vit:.4f}       | {'🟥 FAKE' if score_vit > 0.5 else '🟩 REAL'}")
#         print(f" {'OpenAI CLIP':<18} | {score_clip:.4f}       | {'🟥 FAKE' if score_clip > 0.5 else '🟩 REAL'}")
#         print(f" {'Simple (OpenCV)':<18} | {score_simple:.4f}       | {'🟥 FAKE' if score_simple > 0.5 else '🟩 REAL'}")
#         print("-" * 65)
#         print(f" 🏆 ENSEMBLE RESULT:  {final_score:.4f} -> {verdict} ({confidence}%)")
#         print("="*65 + "\n")

#         return jsonify({
#             "verdict": verdict, 
#             "confidence": confidence,
#             "details": {
#                 "EfficientNet": score_eff,
#                 "ViT": score_vit,
#                 "OpenAI CLIP": score_clip,
#                 "Simple (OpenCV)": score_simple
#             }
#         })

#     except Exception as e:
#         print(f"❌ Error: {e}")
#         return jsonify({"error": str(e)}), 500

# if __name__ == '__main__':
#     # ✅ Port 5000 requested
#     app.run(port=5000, debug=True)
from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io
import base64
import numpy as np
import cv2
import traceback

app = Flask(__name__)
CORS(app)

# --- ตัวแปร Global (ยังไม่โหลดโมเดล) ---
model = None
class_names = ['control', 'dementia']
device = torch.device("cpu")

# --- ฟังก์ชันโหลดโมเดล (จะถูกเรียกเมื่อจำเป็นเท่านั้น) ---
def get_model():
    global model
    if model is not None:
        return model # ถ้าโหลดแล้ว ก็คืนค่าเลย (ไม่ต้องโหลดซ้ำ)
    
    print("⏳ Lazy Loading: Starting to load AI Model...")
    try:
        # สร้างโครงสร้าง (ใช้ ResNet-18 หรือ 101 ตามไฟล์ที่คุณมี)
        # ** ถ้าคุณใช้ไฟล์ resnet18_model.pth ให้ใช้ models.resnet18() **
        # ** ถ้าใช้ไฟล์ best_model.pth (ตัวเก่า) ให้ใช้ models.resnet101() **
        target_model = models.resnet18() 
        
        num_ftrs = target_model.fc.in_features
        target_model.fc = nn.Linear(num_ftrs, 2)
        
        # โหลด Weight
        target_model.load_state_dict(torch.load('resnet18_model.pth', map_location=device))
        target_model.eval()
        
        model = target_model
        print("✅ AI Model loaded successfully!")
        return model
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return None

# --- เตรียมฟังก์ชันแปลงภาพ ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- Route หน้าแรก (Health Check) ---
@app.route('/', methods=['GET'])
def home():
    return "<h1>✅ AI Service is Online!</h1><p>Model will load on first request.</p>"

# --- Route วิเคราะห์ ---
@app.route('/analyze', methods=['POST'])
def analyze_drawing():
    print("\n--- New Request Received ---")

    # 1. เรียกฟังก์ชันโหลดโมเดล (ถ้ายังไม่โหลด มันจะโหลดตอนนี้)
    current_model = get_model()

    if current_model is None:
        return jsonify({"error": "Failed to load AI Model on server."}), 500

    try:
        # รับข้อมูล
        data = request.json
        image_base64 = data.get('image_base64', '')

        if "," in image_base64:
            _, encoded = image_base64.split(",", 1)
        else:
            encoded = image_base64
        
        image_bytes = base64.b64decode(encoded)
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

        # Preprocessing
        cv_image = np.array(image)
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
        blurred = cv2.GaussianBlur(cv_image, (5, 5), 0)
        final_image = Image.fromarray(cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB))

        # Inference
        image_tensor = transform(final_image).unsqueeze(0)

        with torch.no_grad():
            outputs = current_model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, preds = torch.max(probabilities, 1)
            
            prediction = class_names[preds[0]]
            confidence_score = round(confidence.item() * 100, 2)

        print(f"✅ Result: {prediction} ({confidence_score}%)")

        return jsonify({
            "prediction": prediction,
            "confidence": confidence_score
        })

    except Exception as e:
        print("❌ Error:", traceback.format_exc())
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
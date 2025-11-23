from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image, ImageOps # เพิ่ม ImageOps
import io
import base64
import numpy as np
import cv2
import traceback
import os # เพิ่ม os

app = Flask(__name__)
CORS(app)

print("--- STARTUP: Loading AI Model ---")
model = None
class_names = ['control', 'dementia']

try:
    device = torch.device("cpu")
    model = models.resnet18()
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    model.load_state_dict(torch.load('resnet18_model.pth', map_location=device))
    model.eval()
    print("✅ AI Model loaded successfully!")
except Exception as e:
    print(f"❌ Critical Error: {e}")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

@app.route('/analyze', methods=['POST'])
def analyze_drawing():
    print("\n--- New Request ---")

    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    try:
        data = request.json
        image_base64 = data.get('image_base64', '')

        if "," in image_base64:
            _, encoded = image_base64.split(",", 1)
        else:
            encoded = image_base64
        
        image_bytes = base64.b64decode(encoded)
        
        # 1. เปิดภาพ (RGBA รองรับความโปร่งใส)
        image = Image.open(io.BytesIO(image_bytes)).convert('RGBA')

        # ⭐ แก้ไขจุดตาย: ถมพื้นหลังให้เป็นสีขาว (ถ้าโปร่งใส AI จะมองไม่เห็น)
        background = Image.new('RGBA', image.size, (255, 255, 255))
        alpha_composite = Image.alpha_composite(background, image)
        image_rgb = alpha_composite.convert('RGB') # แปลงกลับเป็น RGB

        # 2. Preprocessing (OpenCV)
        cv_image = np.array(image_rgb)
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
        
        # เพิ่มเส้นให้หนาขึ้นอีกนิด (AI ชอบเส้นชัดๆ)
        kernel = np.ones((3,3), np.uint8)
        cv_image = cv2.erode(cv_image, kernel, iterations=1) 
        
        blurred = cv2.GaussianBlur(cv_image, (5, 5), 0)
        final_image = Image.fromarray(cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB))

        # 📸 DEBUG: บันทึกภาพที่ AI เห็น ลงในโฟลเดอร์ backend
        final_image.save("debug_ai_input.png")
        print("📸 Saved debug image to: backend/debug_ai_input.png (Check this file!)")

        # 3. Inference
        image_tensor = transform(final_image).unsqueeze(0)

        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, preds = torch.max(probabilities, 1)
            
            prediction = class_names[preds[0]]
            confidence_score = round(confidence.item() * 100, 2)
            
            # Print ค่า Raw Logits ออกมาดูด้วยว่า AI มั่นใจแค่ไหน
            print(f"📊 Raw Probabilities: {probabilities}")

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
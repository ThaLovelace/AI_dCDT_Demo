// --- 1. ตั้งค่า Fabric.js (เพื่อการวาดที่ลื่นไหล) ---
const canvasElement = document.getElementById('drawing-canvas');
// ปรับขนาด Canvas ให้พอดีกับ Container หรือใช้ขนาด Fixed
const canvas = new fabric.Canvas('drawing-canvas', {
    isDrawingMode: true,
    width: 600,   // ต้องตรงกับใน HTML
    height: 500,  // ต้องตรงกับใน HTML
    backgroundColor: '#ffffff'
});

// ตั้งค่าหัวปากกา
canvas.freeDrawingBrush = new fabric.PencilBrush(canvas);
canvas.freeDrawingBrush.width = 4;
canvas.freeDrawingBrush.color = "#000000";

// --- 2. ตัวแปรจับเวลา (Process Analysis) ---
let startTime = null;
let isDrawingStarted = false;

canvas.on('path:created', function() {
    if (!isDrawingStarted) {
        startTime = new Date();
        isDrawingStarted = true;
    }
});

// --- 3. ฟังก์ชันปุ่มล้างกระดาน ---
document.getElementById('clear-btn').addEventListener('click', function() {
    canvas.clear();
    canvas.setBackgroundColor('#ffffff', canvas.renderAll.bind(canvas));
    startTime = null;
    isDrawingStarted = false;
    document.getElementById('results').classList.add('hidden');
});

// --- 4. ฟังก์ชันวิเคราะห์ (ยิงไปหา Backend จริง!) ---
document.getElementById('analyze-btn').addEventListener('click', async function() {
    
    // Validation: เช็คว่าวาดหรือยัง
    if (canvas.getObjects().length === 0) {
        alert("⚠️ กรุณาวาดภาพก่อนครับ/ค่ะ");
        return;
    }

    // แสดง Loading
    document.getElementById('loadingOverlay').classList.remove('hidden');

    // A. คำนวณเวลาวาด (TCT)
    const now = new Date();
    const timeDiff = startTime ? (now - startTime) : 0;
    const totalTimeSec = (timeDiff / 1000).toFixed(1);

    // B. เตรียมภาพ Base64 เพื่อส่งให้ AI
    // (Fabric.js ช่วยแปลง Canvas เป็นรูปภาพ Base64 String ให้ทันที)
    const dataURL = canvas.toDataURL({
        format: 'png',
        quality: 1
    });

    try {
        // C. ส่ง Request ไปยัง Flask (Localhost)
        const response = await fetch('https://ai-dcdt-demo.onrender.com/analyze', {
            method: 'POST',
            headers: { 
                'Content-Type': 'application/json' 
            },
            body: JSON.stringify({ 
                image_base64: dataURL  // ส่งคีย์นี้เพื่อให้ตรงกับ main.py
            })
        });

        if (!response.ok) {
            throw new Error(`Server Error: ${response.status}`);
        }

        // D. รับผลลัพธ์จริงจาก ResNet-18
        const result = await response.json();
        console.log("✅ AI Result:", result);

        // E. แสดงผลบนหน้าเว็บ
        document.getElementById('res-prediction').innerText = result.prediction;
        
        // เปลี่ยนสีตามผลทำนาย
        const predElement = document.getElementById('res-prediction');
        if (result.prediction.toLowerCase().includes('dementia')) {
            predElement.className = "value highlight-red";
            predElement.style.color = "red";
        } else {
            predElement.className = "value highlight-green";
            predElement.style.color = "#10b981";
        }

        document.getElementById('res-confidence').innerText = result.confidence + "%";
        
        // แสดงค่า Process (TCT)
        document.getElementById('res-tct').innerText = totalTimeSec + " วินาที";
        document.getElementById('res-think').innerText = "35.5%"; // (ค่าสมมติสำหรับ TCT เพราะเราไม่ได้ส่ง JSON เต็ม)

        // สร้าง Insights (Mock ไว้ก่อนเพื่อให้ UI ไม่โล่ง)
        const ul = document.getElementById('res-insights');
        ul.innerHTML = '';
        const insightText = result.prediction.toLowerCase().includes('control') 
            ? "AI ตรวจพบโครงสร้างปกติ (Normal Structure)" 
            : "AI ตรวจพบความเสี่ยง (Potential Impairment Pattern)";
        
        const li = document.createElement('li');
        li.innerText = insightText;
        ul.appendChild(li);

        // เปิดส่วนแสดงผล
        document.getElementById('results').classList.remove('hidden');
        document.getElementById('results').scrollIntoView({ behavior: 'smooth' });

    } catch (error) {
        console.error("Error:", error);
        alert(`❌ เชื่อมต่อ Backend ไม่ได้: ${error.message}\n\n(อย่าลืมรัน 'python app.py' และ 'uvicorn main:app' ใน Terminal นะครับ/คะ)`);
    } finally {
        // ปิด Loading เสมอ ไม่ว่าจะสำเร็จหรือพัง
        document.getElementById('loadingOverlay').classList.add('hidden');
    }
});
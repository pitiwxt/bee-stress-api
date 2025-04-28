# ====== 1. Import Libraries ======
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import numpy as np
import joblib
from tensorflow import keras
import random
import os

# ====== 2. Load Model and Scaler ======
model = keras.models.load_model('best_finetuned_model.keras')
scaler = joblib.load('scaler.pkl')

# ====== 3. Create FastAPI app ======
app = FastAPI()

# ✅ Mount Static Folder
app.mount("/static", StaticFiles(directory="static"), name="static")

# ====== 4. Define Input Data Format ======
class SensorData(BaseModel):
    bw: float
    sc: float
    fre: float

# ====== 5.0 Root Path ======
@app.get("/")
async def root():
    return FileResponse("static/index.html")

# ====== 5.1 API Route for Predicting Stress ======
@app.post("/predict_stress")
async def predict_stress(data: SensorData):
    fre_minus_sc = data.fre - data.sc
    fre_div_bw = data.fre / (data.bw + 1e-6)
    sc_div_bw = data.sc / (data.bw + 1e-6)
    
    features = np.array([[data.fre, data.sc, data.bw, fre_minus_sc, fre_div_bw, sc_div_bw]])
    features_scaled = scaler.transform(features)

    pred_prob = model.predict(features_scaled)[0][0]
    stress_level = int(pred_prob >= 0.5)

    return {
        "stress_probability": float(pred_prob),
        "stress_detected": bool(stress_level)
    }

# ====== 5.2 API Real-time Data ======
@app.get("/realtime_data")
async def realtime_data():
    bw = random.uniform(5000, 7000)
    sc = random.uniform(400, 500)
    fre = random.uniform(410, 470)

    fre_minus_sc = fre - sc
    fre_div_bw = fre / (bw + 1e-6)
    sc_div_bw = sc / (bw + 1e-6)

    features = np.array([[fre, sc, bw, fre_minus_sc, fre_div_bw, sc_div_bw]])
    features_scaled = scaler.transform(features)

    pred_prob = model.predict(features_scaled)[0][0]

    return {
        "bw": bw,
        "sc": sc,
        "fre": fre,
        "stress_level": float(pred_prob)
    }

# ====== 6. Run Server ======
if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8080))
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=port)

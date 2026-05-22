from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import shutil
import os
import uuid
import numpy as np
import librosa
import tensorflow as tf

# 创建临时文件夹
TEMP_DIR = os.path.join(os.getcwd(), "temp")
os.makedirs(TEMP_DIR, exist_ok=True)

# 全局变量
MODEL = None
CLASSES = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global MODEL, CLASSES
    print("🚀 Loading model...")
    MODEL = tf.keras.models.load_model("sound_class_model_mfcc_opt.keras")
    CLASSES = np.load("classes.npy", allow_pickle=True)
    print(f"✅ Model ready! {len(CLASSES)} classes")
    yield
    print("🛑 Shutting down...")

app = FastAPI(title="AI Sound Detection API", version="1.0", lifespan=lifespan)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def extract_features(file_path, sr=22050, n_mfcc=40, max_len=128, n_mels=128):
    """特征提取（与训练时一致）"""
    y, sr = librosa.load(file_path, sr=sr, duration=5.0)
    
    if len(y) < sr * 5:
        y = np.pad(y, (0, sr * 5 - len(y)))
    else:
        y = y[:sr * 5]
    
    # MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    mfcc_features = np.concatenate([mfcc, delta, delta2], axis=0)
    
    # Mel 频谱图
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    mel_spec_db = librosa.power_to_db(mel_spec)
    
    # Padding
    if mfcc_features.shape[1] < max_len:
        pad = max_len - mfcc_features.shape[1]
        mfcc_features = np.pad(mfcc_features, ((0, 0), (0, pad)))
    else:
        mfcc_features = mfcc_features[:, :max_len]
    
    if mel_spec_db.shape[1] < max_len:
        pad = max_len - mel_spec_db.shape[1]
        mel_spec_db = np.pad(mel_spec_db, ((0, 0), (0, pad)))
    else:
        mel_spec_db = mel_spec_db[:, :max_len]
    
    combined = np.concatenate([mfcc_features, mel_spec_db], axis=0)
    combined = combined.T[np.newaxis, ..., np.newaxis]
    return combined.astype(np.float32)

@app.get("/")
def home():
    return {
        "status": "running",
        "classes": list(CLASSES) if CLASSES is not None else []
    }

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/detect_sound")
async def detect_sound(file: UploadFile = File(...)):
    if MODEL is None:
        return {"error": "Model not loaded"}
    
    temp_path = os.path.join(TEMP_DIR, f"{uuid.uuid4().hex}_{file.filename}")
    
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        features = extract_features(temp_path)
        predictions = MODEL.predict(features, verbose=0)
        idx = np.argmax(predictions[0])
        confidence = float(np.max(predictions[0]))
        label = CLASSES[idx]
        
        # Top 3
        top3_idx = np.argsort(predictions[0])[-3:][::-1]
        top3 = [(CLASSES[i], float(predictions[0][i])) for i in top3_idx]
        
        return {
            "label": label,
            "confidence": confidence,
            "top3": top3
        }
    
    except Exception as e:
        return {"error": str(e)}
    
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port)
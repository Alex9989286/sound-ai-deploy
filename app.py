# # from fastapi import FastAPI, UploadFile, File
# # from fastapi.middleware.cors import CORSMiddleware
# # from contextlib import asynccontextmanager
# # import shutil
# # import os
# # import uuid
# # import numpy as np
# # import librosa
# # import tensorflow as tf

# # # 创建临时文件夹
# # TEMP_DIR = os.path.join(os.getcwd(), "temp")
# # os.makedirs(TEMP_DIR, exist_ok=True)

# # # 全局变量
# # MODEL = None
# # CLASSES = None

# # @asynccontextmanager
# # async def lifespan(app: FastAPI):
# #     global MODEL, CLASSES
# #     print("🚀 Loading model...")
# #     MODEL = tf.keras.models.load_model("sound_class_model_mfcc_opt.keras")
# #     CLASSES = np.load("classes.npy", allow_pickle=True)
# #     print(f"✅ Model ready! {len(CLASSES)} classes")
# #     yield
# #     print("🛑 Shutting down...")

# # app = FastAPI(title="AI Sound Detection API", version="1.0", lifespan=lifespan)

# # # CORS
# # app.add_middleware(
# #     CORSMiddleware,
# #     allow_origins=["*"],
# #     allow_credentials=True,
# #     allow_methods=["*"],
# #     allow_headers=["*"],
# # )

# # def extract_features(file_path, sr=22050, n_mfcc=40, max_len=128, n_mels=128):
# #     """特征提取（与训练时一致）"""
# #     y, sr = librosa.load(file_path, sr=sr, duration=5.0)
    
# #     if len(y) < sr * 5:
# #         y = np.pad(y, (0, sr * 5 - len(y)))
# #     else:
# #         y = y[:sr * 5]
    
# #     # MFCC
# #     mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
# #     delta = librosa.feature.delta(mfcc)
# #     delta2 = librosa.feature.delta(mfcc, order=2)
# #     mfcc_features = np.concatenate([mfcc, delta, delta2], axis=0)
    
# #     # Mel 频谱图
# #     mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
# #     mel_spec_db = librosa.power_to_db(mel_spec)
    
# #     # Padding
# #     if mfcc_features.shape[1] < max_len:
# #         pad = max_len - mfcc_features.shape[1]
# #         mfcc_features = np.pad(mfcc_features, ((0, 0), (0, pad)))
# #     else:
# #         mfcc_features = mfcc_features[:, :max_len]
    
# #     if mel_spec_db.shape[1] < max_len:
# #         pad = max_len - mel_spec_db.shape[1]
# #         mel_spec_db = np.pad(mel_spec_db, ((0, 0), (0, pad)))
# #     else:
# #         mel_spec_db = mel_spec_db[:, :max_len]
    
# #     combined = np.concatenate([mfcc_features, mel_spec_db], axis=0)
# #     combined = combined.T[np.newaxis, ..., np.newaxis]
# #     return combined.astype(np.float32)

# # @app.get("/")
# # def home():
# #     return {
# #         "status": "running",
# #         "classes": list(CLASSES) if CLASSES is not None else []
# #     }

# # @app.get("/health")
# # def health():
# #     return {"status": "ok"}

# # @app.post("/detect_sound")
# # async def detect_sound(file: UploadFile = File(...)):
# #     if MODEL is None:
# #         return {"error": "Model not loaded"}
    
# #     temp_path = os.path.join(TEMP_DIR, f"{uuid.uuid4().hex}_{file.filename}")
    
# #     try:
# #         with open(temp_path, "wb") as buffer:
# #             shutil.copyfileobj(file.file, buffer)
        
# #         features = extract_features(temp_path)
# #         predictions = MODEL.predict(features, verbose=0)
# #         idx = np.argmax(predictions[0])
# #         confidence = float(np.max(predictions[0]))
# #         label = CLASSES[idx]
        
# #         # Top 3
# #         top3_idx = np.argsort(predictions[0])[-3:][::-1]
# #         top3 = [(CLASSES[i], float(predictions[0][i])) for i in top3_idx]
        
# #         return {
# #             "label": label,
# #             "confidence": confidence,
# #             "top3": top3
# #         }
    
# #     except Exception as e:
# #         return {"error": str(e)}
    
# #     finally:
# #         if os.path.exists(temp_path):
# #             os.remove(temp_path)

# # if __name__ == "__main__":
# #     import uvicorn
# #     port = int(os.environ.get("PORT", 8000))
# #     uvicorn.run("app:app", host="0.0.0.0", port=port)


# from fastapi import FastAPI, UploadFile, File
# from fastapi.middleware.cors import CORSMiddleware
# from contextlib import asynccontextmanager
# import shutil
# import os
# import uuid
# import numpy as np
# import librosa
# import tensorflow as tf

# # ============================================
# # 配置
# # ============================================
# TEMP_DIR = os.path.join(os.getcwd(), "temp")
# os.makedirs(TEMP_DIR, exist_ok=True)

# MODEL = None
# CLASSES = None

# # ============================================
# # 加载模型 - 使用 SavedModel 格式 (最可靠)
# # ============================================
# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     global MODEL, CLASSES
#     print("🚀 Loading model from SavedModel...")

#     # 直接加载整个 SavedModel 文件夹
#     # 这种方式同时加载了模型的结构和权重，不会有任何自定义对象的序列化问题
#     MODEL = tf.saved_model.load("saved_model")
    
#     # 获取模型的推断函数
#     # 注意：SavedModel 加载后通常需要 .signatures 来调用，或者可以直接作为可调用对象
#     # 为了保持你原有的 MODEL.predict 调用方式，我们可以取默认的 serving 函数
#     MODEL = MODEL.signatures['serving_default']
    
#     print("✅ Model loaded from SavedModel.")

#     # 加载类别
#     CLASSES = np.load("classes.npy", allow_pickle=True)
#     print(f"✅ Classes loaded: {len(CLASSES)} classes")
#     print(f"   Classes: {list(CLASSES)}")

#     yield
#     print("🛑 Shutting down...")

# app = FastAPI(title="AI Sound Detection API", version="1.0", lifespan=lifespan)

# # CORS
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # ============================================
# # 特征提取 (与训练时一致)
# # ============================================
# def extract_features(file_path, sr=22050, n_mfcc=40, max_len=128, n_mels=128):
#     """提取特征，返回 float32"""
#     y, sr = librosa.load(file_path, sr=sr, duration=5.0)
    
#     if len(y) < sr * 5:
#         y = np.pad(y, (0, sr * 5 - len(y)))
#     else:
#         y = y[:sr * 5]
    
#     mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
#     delta = librosa.feature.delta(mfcc)
#     delta2 = librosa.feature.delta(mfcc, order=2)
#     mfcc_features = np.concatenate([mfcc, delta, delta2], axis=0)
    
#     mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
#     mel_spec_db = librosa.power_to_db(mel_spec)
    
#     if mfcc_features.shape[1] < max_len:
#         pad = max_len - mfcc_features.shape[1]
#         mfcc_features = np.pad(mfcc_features, ((0, 0), (0, pad)))
#     else:
#         mfcc_features = mfcc_features[:, :max_len]
    
#     if mel_spec_db.shape[1] < max_len:
#         pad = max_len - mel_spec_db.shape[1]
#         mel_spec_db = np.pad(mel_spec_db, ((0, 0), (0, pad)))
#     else:
#         mel_spec_db = mel_spec_db[:, :max_len]
    
#     combined = np.concatenate([mfcc_features, mel_spec_db], axis=0)
#     combined = combined.T[np.newaxis, ..., np.newaxis]
#     return combined.astype(np.float32)


# @app.post("/detect_sound")
# async def detect_sound(file: UploadFile = File(...)):
#     if MODEL is None:
#         return {"error": "Model not loaded"}
    
#     temp_path = os.path.join(TEMP_DIR, f"{uuid.uuid4().hex}_{file.filename}")
    
#     try:
#         # 保存上传的文件
#         with open(temp_path, "wb") as buffer:
#             shutil.copyfileobj(file.file, buffer)
        
#         # 提取特征
#         features = extract_features(temp_path)
        
#         # 使用 SavedModel 进行推理
#         # 注意：SavedModel 的 serving_default 函数要求输入是一个字典
#         predictions = MODEL(tf.constant(features))
#         # 输出也是一个字典，键通常是 'output_0' 或 'dense_1'，需要从中取出结果
#         pred_values = list(predictions.values())[0].numpy()
        
#         idx = np.argmax(pred_values[0])
#         confidence = float(np.max(pred_values[0]))
#         label = CLASSES[idx]
        
#         # Top 3
#         top3_idx = np.argsort(pred_values[0])[-3:][::-1]
#         top3 = [(CLASSES[i], float(pred_values[0][i])) for i in top3_idx]
        
#         return {
#             "label": label,
#             "confidence": confidence,
#             "top3": top3
#         }
    
#     except Exception as e:
#         return {"error": str(e)}
    
#     finally:
#         if os.path.exists(temp_path):
#             os.remove(temp_path)

# @app.get("/")
# def home():
#     return {
#         "status": "running",
#         "classes": list(CLASSES) if CLASSES is not None else []
#     }

# @app.get("/health")
# def health():
#     return {"status": "ok"}

# if __name__ == "__main__":
#     import uvicorn
#     port = int(os.environ.get("PORT", 8000))
#     uvicorn.run("app:app", host="0.0.0.0", port=port)

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import shutil
import os
import uuid
import numpy as np
import librosa
import tensorflow as tf
import asyncio
from concurrent.futures import ThreadPoolExecutor
import time

# ============================================
# 配置
# ============================================
TEMP_DIR = os.path.join(os.getcwd(), "temp")
os.makedirs(TEMP_DIR, exist_ok=True)

# 处理超时时间（秒）- 超过此时间则放弃
PROCESS_TIMEOUT = 12

# 线程池（用于隔离同步任务）
executor = ThreadPoolExecutor(max_workers=2)

MODEL = None
CLASSES = None

# ============================================
# 加载模型 - 使用 SavedModel 格式 (最可靠)
# ============================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    global MODEL, CLASSES
    print("🚀 Loading model from SavedModel...")

    MODEL = tf.saved_model.load("saved_model")
    MODEL = MODEL.signatures['serving_default']
    
    print("✅ Model loaded from SavedModel.")

    CLASSES = np.load("classes.npy", allow_pickle=True)
    print(f"✅ Classes loaded: {len(CLASSES)} classes")
    print(f"   Classes: {list(CLASSES)}")

    yield
    print("🛑 Shutting down...")
    executor.shutdown(wait=False)

app = FastAPI(title="AI Sound Detection API", version="1.0", lifespan=lifespan)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================
# 特征提取 (与训练时一致)
# ============================================
def extract_features(file_path, sr=22050, n_mfcc=40, max_len=128, n_mels=128):
    """提取特征，返回 float32"""
    y, sr = librosa.load(file_path, sr=sr, duration=5.0)
    
    if len(y) < sr * 5:
        y = np.pad(y, (0, sr * 5 - len(y)))
    else:
        y = y[:sr * 5]
    
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    mfcc_features = np.concatenate([mfcc, delta, delta2], axis=0)
    
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    mel_spec_db = librosa.power_to_db(mel_spec)
    
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


# ============================================
# 同步处理函数（在线程池中运行）
# ============================================
def sync_process_audio(file_path: str):
    """同步处理音频（在线程池中执行）"""
    try:
        features = extract_features(file_path)
        predictions = MODEL(tf.constant(features))
        pred_values = list(predictions.values())[0].numpy()
        
        idx = np.argmax(pred_values[0])
        confidence = float(np.max(pred_values[0]))
        label = CLASSES[idx]
        
        top3_idx = np.argsort(pred_values[0])[-3:][::-1]
        top3 = [(CLASSES[i], float(pred_values[0][i])) for i in top3_idx]
        
        return {
            "label": label,
            "confidence": confidence,
            "top3": top3
        }
    except Exception as e:
        return {"error": str(e), "label": "unknown", "confidence": 0.0}


# ============================================
# 检测端点（带超时控制）
# ============================================
@app.post("/detect_sound")
async def detect_sound(file: UploadFile = File(...)):
    if MODEL is None:
        return {"error": "Model not loaded"}
    
    temp_path = os.path.join(TEMP_DIR, f"{uuid.uuid4().hex}_{file.filename}")
    
    try:
        # 保存上传的文件
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # 在线程池中运行同步任务，并设置超时
        try:
            # 使用 asyncio.to_thread 将同步任务放到线程池
            result = await asyncio.wait_for(
                asyncio.to_thread(sync_process_audio, temp_path),
                timeout=PROCESS_TIMEOUT
            )
            
            # 检查是否有错误
            if "error" in result and result["error"]:
                return {"error": result["error"], "label": "unknown", "confidence": 0.0}
            
            return result
            
        except asyncio.TimeoutError:
            print(f"⚠️ Processing timeout after {PROCESS_TIMEOUT} seconds")
            return {
                "error": "Processing timeout",
                "label": "unknown",
                "confidence": 0.0,
                "message": f"Audio processing exceeded {PROCESS_TIMEOUT} seconds"
            }
    
    except Exception as e:
        print(f"❌ Error: {e}")
        return {"error": str(e), "label": "unknown", "confidence": 0.0}
    
    finally:
        # 清理临时文件
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass


@app.get("/")
def home():
    return {
        "status": "running",
        "classes": list(CLASSES) if CLASSES is not None else []
    }


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port)